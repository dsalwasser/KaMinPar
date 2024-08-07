/*******************************************************************************
 * Deep multilevel graph partitioning scheme.
 *
 * @file:   deep_multilevel.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#include "kaminpar-shm/partitioning/deep/deep_multilevel.h"

#include "kaminpar-shm/coarsening/max_cluster_weights.h"
#include "kaminpar-shm/factories.h"
#include "kaminpar-shm/partitioning/debug.h"
#include "kaminpar-shm/partitioning/deep/async_initial_partitioning.h"
#include "kaminpar-shm/partitioning/deep/sync_initial_partitioning.h"
#include "kaminpar-shm/partitioning/helper.h"
#include "kaminpar-shm/partitioning/partition_utils.h"

#include "kaminpar-common/console_io.h"
#include "kaminpar-common/heap_profiler.h"

namespace kaminpar::shm {
namespace {
SET_DEBUG(false);
} // namespace

using namespace partitioning;

DeepMultilevelPartitioner::DeepMultilevelPartitioner(
    const Graph &input_graph, const Context &input_ctx
)
    : _input_graph(input_graph),
      _input_ctx(input_ctx),
      _current_p_ctx(input_ctx.partition),
      _coarsener(factory::create_coarsener(input_ctx)),
      _refiner(factory::create_refiner(input_ctx)),
      _bipartitioner_pool(_input_ctx) {
  _coarsener->initialize(&_input_graph);
}

PartitionedGraph DeepMultilevelPartitioner::partition() {
  cio::print_delimiter("Partitioning");

  const Graph *c_graph = coarsen();
  PartitionedGraph p_graph = initial_partition(c_graph);

  bool refined = false;
  p_graph = uncoarsen(std::move(p_graph), refined);
  if (!refined || p_graph.k() < _input_ctx.partition.k) {
    LOG;
    LOG << "Toplevel:";
    LOG << "  Number of nodes: " << p_graph.n() << " | Number of edges: " << p_graph.m();

    if (!refined) {
      refine(p_graph);
    }
    if (p_graph.k() < _input_ctx.partition.k) {
      extend_partition(p_graph, _input_ctx.partition.k);
      refine(p_graph);
    }
  }

  return p_graph;
}

PartitionedGraph DeepMultilevelPartitioner::uncoarsen_once(PartitionedGraph p_graph) {
  return partitioning::uncoarsen_once(
      _coarsener.get(), std::move(p_graph), _current_p_ctx, _input_ctx.partition
  );
}

void DeepMultilevelPartitioner::refine(PartitionedGraph &p_graph) {
  SCOPED_HEAP_PROFILER("Refinement");

  // If requested, dump the current partition to disk before refinement ...
  debug::dump_partition_hierarchy(p_graph, _coarsener->level(), "pre-refinement", _input_ctx);

  LOG << "  Running refinement on " << p_graph.k() << " blocks";
  partitioning::refine(_refiner.get(), p_graph, _current_p_ctx);

  if (_print_metrics) {
    SCOPED_TIMER("Partition metrics");
    LOG << "    Cut:       " << metrics::edge_cut(p_graph);
    LOG << "    Imbalance: " << metrics::imbalance(p_graph);
    LOG << "    Feasible:  " << metrics::is_feasible(p_graph, _current_p_ctx);
  }

  // ... and dump it after refinement.
  debug::dump_partition_hierarchy(p_graph, _coarsener->level(), "post-refinement", _input_ctx);
}

void DeepMultilevelPartitioner::extend_partition(PartitionedGraph &p_graph, const BlockID k_prime) {
  SCOPED_HEAP_PROFILER("Extending partition");
  LOG << "  Extending partition from " << p_graph.k() << " blocks to " << k_prime << " blocks";
  partitioning::extend_partition(
      p_graph,
      k_prime,
      _input_ctx,
      _current_p_ctx,
      _subgraph_memory,
      _tmp_extraction_mem_pool_ets,
      _bipartitioner_pool,
      _input_ctx.parallel.num_threads
  );

  if (_print_metrics) {
    SCOPED_TIMER("Partition metrics");
    LOG << "    Cut:       " << metrics::edge_cut(p_graph);
    LOG << "    Imbalance: " << metrics::imbalance(p_graph);
  }
}

PartitionedGraph DeepMultilevelPartitioner::uncoarsen(PartitionedGraph p_graph, bool &refined) {
  SCOPED_HEAP_PROFILER("Uncoarsening");

  while (!_coarsener->empty()) {
    LOG;
    LOG << "Uncoarsening -> Level " << (_coarsener->level() - 1);

    p_graph = uncoarsen_once(std::move(p_graph));

    LOG << "  Number of nodes: " << p_graph.n() << " | Number of edges: " << p_graph.m();

    refine(p_graph);
    refined = true;

    const BlockID desired_k = partitioning::compute_k_for_n(p_graph.n(), _input_ctx);
    if (p_graph.k() < desired_k) {
      extend_partition(p_graph, desired_k);
      refined = false;
    }
  }

  return p_graph;
}

const Graph *DeepMultilevelPartitioner::coarsen() {
  SCOPED_HEAP_PROFILER("Coarsening");

  const Graph *c_graph = &_input_graph;
  NodeID prev_c_graph_n = c_graph->n();
  EdgeID prev_c_graph_m = c_graph->m();
  NodeWeight prev_c_graph_total_node_weight = c_graph->total_node_weight();
  bool shrunk = true;

  bool search_subgraph_memory_size = true;
  NodeID subgraph_memory_n;
  EdgeID subgraph_memory_m;

  while (shrunk && c_graph->n() > initial_partitioning_threshold()) {
    // If requested, dump graph before each coarsening step + after coarsening
    // converged. This way, we also have a dump of the (reordered) input graph,
    // which makes it easier to use the final partition (before reordering it).
    // We dump the coarsest graph in ::initial_partitioning().
    debug::dump_graph_hierarchy(*c_graph, _coarsener->level(), _input_ctx);

    // Store the size of the previous coarse graph, so that we can pre-allocate _subgraph_memory
    // if we need it for this graph (see below)
    prev_c_graph_n = c_graph->n();
    prev_c_graph_m = c_graph->m();
    prev_c_graph_total_node_weight = c_graph->total_node_weight();

    // Build next coarse graph
    shrunk = partitioning::coarsen_once(_coarsener.get(), c_graph, _current_p_ctx);
    c_graph = &_coarsener->current();

    // _subgraph_memory stores the block-induced subgraphs of the partitioned graph during recursive
    // bipartitioning
    // To avoid repeated allocation, we pre-allocate the memory during coarsening for the largest
    // coarse graph for which we still need recursive bipartitioning
    if (search_subgraph_memory_size &&
        partitioning::compute_k_for_n(c_graph->n(), _input_ctx) < _input_ctx.partition.k) {
      search_subgraph_memory_size = false;
      subgraph_memory_n = prev_c_graph_n;
      subgraph_memory_m = prev_c_graph_m;
    }

    // Print some metrics for the coarse graphs
    LOG << "Coarsening -> Level " << _coarsener->level();
    LOG << "  Number of nodes: " << c_graph->n() << " | Number of edges: " << c_graph->m();
    LLOG << "  Maximum node weight: " << c_graph->max_node_weight() << " ";
    LLOG << "<= "
         << compute_max_cluster_weight<NodeWeight>(
                _input_ctx.coarsening,
                _input_ctx.partition,
                prev_c_graph_n,
                prev_c_graph_total_node_weight
            );
    LOG;
    LOG;
  }

  if (search_subgraph_memory_size) {
    subgraph_memory_n = prev_c_graph_n;
    subgraph_memory_m = prev_c_graph_m;
  }
  _subgraph_memory.resize(subgraph_memory_n, _input_ctx.partition.k, subgraph_memory_m, true, true);

  TIMED_SCOPE("Coarsening") {
    _coarsener->release_allocated_memory();
  };

  if (shrunk) {
    LOG << "==> Coarsening terminated with less than " << initial_partitioning_threshold()
        << " nodes.";
    LOG;
  } else {
    LOG << "==> Coarsening converged.";
    LOG;
  }

  return c_graph;
}

NodeID DeepMultilevelPartitioner::initial_partitioning_threshold() {
  if (partitioning::parallel_ip_mode(_input_ctx.partitioning.deep_initial_partitioning_mode)) {
    return _input_ctx.parallel.num_threads * _input_ctx.coarsening.contraction_limit; // p * C
  } else {
    return 2 * _input_ctx.coarsening.contraction_limit; // 2 * C
  }
}

PartitionedGraph DeepMultilevelPartitioner::initial_partition(const Graph *graph) {
  SCOPED_HEAP_PROFILER("Initial partitioning");
  SCOPED_TIMER("Initial partitioning scheme");
  LOG << "Initial partitioning:";

  // If requested, dump the coarsest graph to disk. Note that in the context of
  // deep multilevel, this is not actually the coarsest graph, but rather the
  // coarsest graph before splitting PEs and duplicating the graph.
  // Disable worker splitting with --p-deep-initial-partitioning-mode=sequential to obtain coarser
  // graphs.
  debug::dump_coarsest_graph(*graph, _input_ctx);
  debug::dump_graph_hierarchy(*graph, _coarsener->level(), _input_ctx);

  // Since timers are not multi-threaded, we disable them during (parallel)
  // initial partitioning.
  DISABLE_TIMERS();
  PartitionedGraph p_graph = [&] {
    switch (_input_ctx.partitioning.deep_initial_partitioning_mode) {
    case InitialPartitioningMode::SEQUENTIAL:
      return partitioning::bipartition(
          graph, _input_ctx.partition.k, _input_ctx, _bipartitioner_pool, true
      );

    case InitialPartitioningMode::SYNCHRONOUS_PARALLEL:
      return SyncInitialPartitioner(_input_ctx, _bipartitioner_pool, _tmp_extraction_mem_pool_ets)
          .partition(_coarsener.get(), _current_p_ctx);

    case InitialPartitioningMode::ASYNCHRONOUS_PARALLEL:
      return AsyncInitialPartitioner(_input_ctx, _bipartitioner_pool, _tmp_extraction_mem_pool_ets)
          .partition(_coarsener.get(), _current_p_ctx);
    }

    __builtin_unreachable();
  }();
  ENABLE_TIMERS();
  partitioning::update_partition_context(_current_p_ctx, p_graph, _input_ctx.partition.k);

  // Print some metrics for the initial partition.
  LOG << "  Number of blocks: " << p_graph.k();
  if (_print_metrics) {
    SCOPED_TIMER("Partition metrics");
    LOG << "  Cut:              " << metrics::edge_cut(p_graph);
    LOG << "  Imbalance:        " << metrics::imbalance(p_graph);
    LOG << "  Feasible:         " << (metrics::is_feasible(p_graph, _current_p_ctx) ? "yes" : "no");
  }

  // If requested, dump the coarsest partition -- as noted above, this is not
  // actually the coarsest partition when using deep multilevel.
  debug::dump_coarsest_partition(p_graph, _input_ctx);
  debug::dump_partition_hierarchy(p_graph, _coarsener->level(), "post-refinement", _input_ctx);

  return p_graph;
}
} // namespace kaminpar::shm
