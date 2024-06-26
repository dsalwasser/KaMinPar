/*******************************************************************************
 * Utility functions for common operations used by partitioning schemes.
 *
 * @file:   helper.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#include "kaminpar-shm/partitioning/helper.h"

#include "kaminpar-shm/partitioning/partition_utils.h"

#include "kaminpar-common/math.h"

namespace kaminpar::shm::partitioning {
namespace {
SET_DEBUG(false);
SET_STATISTICS_FROM_GLOBAL();
} // namespace

void update_partition_context(
    PartitionContext &current_p_ctx, const PartitionedGraph &p_graph, const BlockID input_k
) {
  current_p_ctx.setup(p_graph.graph());
  current_p_ctx.k = p_graph.k();
  current_p_ctx.block_weights.setup(current_p_ctx, input_k);
}

PartitionedGraph uncoarsen_once(
    Coarsener *coarsener,
    PartitionedGraph p_graph,
    PartitionContext &current_p_ctx,
    const PartitionContext &input_p_ctx
) {
  SCOPED_TIMER("Uncoarsening");

  if (!coarsener->empty()) {
    p_graph = coarsener->uncoarsen(std::move(p_graph));
    update_partition_context(current_p_ctx, p_graph, input_p_ctx.k);
  }

  return p_graph;
}

void refine(Refiner *refiner, PartitionedGraph &p_graph, const PartitionContext &current_p_ctx) {
  SCOPED_TIMER("Refinement");
  refiner->initialize(p_graph);
  refiner->refine(p_graph, current_p_ctx);
}

PartitionedGraph bipartition(
    const Graph *graph,
    const BlockID final_k,
    const Context &input_ctx,
    InitialBipartitionerWorkerPool &initial_bipartitioner_pool,
    const bool partition_lifespan,
    BipartitionTimingInfo *timings
) {
  timer::LocalTimer timer;

  const CSRGraph *csr = dynamic_cast<const CSRGraph *>(graph->underlying_graph());

  // If we work with something other than a CSRGraph, construct a CSR copy to call the initial
  // partitioning code
  // This should only be necessary if the graph is too small for coarsening *and* we are using the
  // compressed mode
  std::unique_ptr<CSRGraph> csr_cpy;
  if (csr == nullptr) {
    DBG << "Bipartitioning a non-CSR graph is not supported by the initial partitioning code: "
           "constructing a CSR-graph copy of the given graph with n="
        << graph->n() << ", m=" << graph->m();
    DBG << "Note: this should only happen when partitioning a very small graph using the "
           "compressed mode";

    csr_cpy = std::make_unique<CSRGraph>(*graph);
    csr = csr_cpy.get();
  }

  timer.reset();
  auto bipartition = [&] {
    InitialMultilevelBipartitioner bipartitioner = initial_bipartitioner_pool.get();
    bipartitioner.init(*csr, final_k);
    auto bipartition =
        bipartitioner.partition(timings ? &(timings->ip_timings) : nullptr).take_raw_partition();

    if (partition_lifespan) {
      StaticArray<BlockID> owned_bipartition(bipartition.size());
      std::copy(bipartition.begin(), bipartition.end(), owned_bipartition.begin());

      initial_bipartitioner_pool.put(std::move(bipartitioner));

      return owned_bipartition;
    } else {
      initial_bipartitioner_pool.put(std::move(bipartitioner));
      return bipartition;
    }
  }();
  if (timings != nullptr) {
    timings->bipartitioner_ms += timer.elapsed();
  }

  timer.reset();
  PartitionedGraph p_graph(PartitionedGraph::seq{}, *graph, 2, std::move(bipartition));
  if (timings != nullptr) {
    timings->graph_init_ms += timer.elapsed();
  }

  return p_graph;
}

void extend_partition_recursive(
    const Graph &graph,
    StaticArray<BlockID> &partition,
    const BlockID b0,
    const BlockID k,
    const BlockID final_k,
    const Context &input_ctx,
    graph::SubgraphMemory &subgraph_memory,
    const graph::SubgraphMemoryStartPosition position,
    TemporarySubgraphMemoryEts &tmp_extraction_mem_pool_ets,
    InitialBipartitionerWorkerPool &bipartitioner_pool,
    BipartitionTimingInfo *timings
) {
  KASSERT(k > 1u);

  PartitionedGraph p_graph =
      bipartition(&graph, final_k, input_ctx, bipartitioner_pool, false, timings);

  timer::LocalTimer timer;

  timer.reset();
  std::array<BlockID, 2> final_ks{0, 0};
  std::array<BlockID, 2> ks{0, 0};
  std::tie(final_ks[0], final_ks[1]) = math::split_integral(final_k);
  std::tie(ks[0], ks[1]) = math::split_integral(k);
  std::array<BlockID, 2> b{b0, b0 + ks[0]};
  if (timings != nullptr)
    timings->misc_ms += timer.elapsed();

  DBG << "bipartitioning graph with weight " << graph.total_node_weight() << " = "
      << p_graph.block_weight(0) << " + " << p_graph.block_weight(1) << " for final k " << final_k
      << " = " << final_ks[0] << " + " << final_ks[1] << ", for total of " << k << " = " << ks[0]
      << " + " << ks[1] << " blocks";

  KASSERT(ks[0] >= 1u);
  KASSERT(ks[1] >= 1u);
  KASSERT(final_ks[0] >= ks[0]);
  KASSERT(final_ks[1] >= ks[1]);
  KASSERT(b[0] < input_ctx.partition.k);
  KASSERT(b[1] < input_ctx.partition.k);

  // Copy p_graph to partition -> replace b0 with b0 or b1
  {
    timer.reset();
    NodeID node = 0;
    for (BlockID &block : partition) {
      block = (block == b0) ? b[p_graph.block(node++)] : block;
    }
    KASSERT(node == p_graph.n());
    if (timings != nullptr)
      timings->copy_ms += timer.elapsed();
  }

  if (k > 2) {
    timer.reset();
    auto extraction = extract_subgraphs_sequential(
        p_graph, final_ks, position, subgraph_memory, tmp_extraction_mem_pool_ets.local()
    );
    const auto &subgraphs = extraction.subgraphs;
    const auto &positions = extraction.positions;
    if (timings != nullptr)
      timings->extract_ms += timer.elapsed();

    for (const std::size_t i : {0, 1}) {
      if (ks[i] <= 1) {
        continue;
      }

      extend_partition_recursive(
          subgraphs[i],
          partition,
          b[i],
          ks[i],
          final_ks[i],
          input_ctx,
          subgraph_memory,
          positions[i],
          tmp_extraction_mem_pool_ets,
          bipartitioner_pool,
          timings
      );
    }
  }
}

void extend_partition(
    PartitionedGraph &p_graph, // stores current k
    const BlockID k_prime,     // extend to this many blocks
    const Context &input_ctx,  // stores input k
    PartitionContext &current_p_ctx,
    graph::SubgraphMemory &subgraph_memory,
    TemporarySubgraphMemoryEts &tmp_extraction_mem_pool_ets,
    InitialBipartitionerWorkerPool &bipartitioner_pool,
    const int num_active_threads
) {
  if (input_ctx.partitioning.min_consecutive_seq_bipartitioning_levels > 0) {
    // Depending on the coarsening level and the deep multilevel implementation, it can occur that
    // this function is called with more threads than blocks in the graph partition. To avoid
    // wasting threads, we only extend the partition a little at first, and then recurse until all
    // threads can work on independent blocks.
    // "min_consecutive_seq_bipartitioning_levels" parameterizes the term "a little": when set to 1,
    // we have the most amount of parallelization, but waste time by re-extracting the block-induced
    // subgraphs from the partitioned graph; larger values do this less often at the cost of wasting
    // more parallel compute resources.
    // @todo change async_initial_partitioning.{cc, h} to make this obsolete ...
    const int factor = 2 << (input_ctx.partitioning.min_consecutive_seq_bipartitioning_levels - 1);
    while (k_prime > factor * p_graph.k() && num_active_threads > p_graph.k()) {
      extend_partition(
          p_graph,
          factor * p_graph.k(),
          input_ctx,
          current_p_ctx,
          subgraph_memory,
          tmp_extraction_mem_pool_ets,
          bipartitioner_pool,
          num_active_threads
      );
    }
  }

  SCOPED_TIMER("Initial partitioning");

  START_HEAP_PROFILER("Extract subgraphs");
  auto extraction = TIMED_SCOPE("Extract subgraphs") {
    return extract_subgraphs(p_graph, input_ctx.partition.k, subgraph_memory);
  };
  STOP_HEAP_PROFILER();
  const auto &subgraphs = extraction.subgraphs;
  const auto &mapping = extraction.node_mapping;
  const auto &positions = extraction.positions;

  START_HEAP_PROFILER("Allocation");
  START_TIMER("Allocation");
  ScalableVector<StaticArray<BlockID>> subgraph_partitions;
  for (const auto &subgraph : subgraphs) {
    subgraph_partitions.emplace_back(subgraph.n());
  }
  STOP_TIMER();
  STOP_HEAP_PROFILER();

  START_HEAP_PROFILER("Bipartitioning");
  START_TIMER("Bipartitioning");
  tbb::enumerable_thread_specific<BipartitionTimingInfo> timings_ets;

  tbb::parallel_for<BlockID>(0, subgraphs.size(), [&](const BlockID b) {
    BipartitionTimingInfo &timing = timings_ets.local();

    const auto &subgraph = subgraphs[b];
    const BlockID final_kb = compute_final_k(b, p_graph.k(), input_ctx.partition.k);

    const BlockID subgraph_k =
        (k_prime == input_ctx.partition.k) ? final_kb : k_prime / p_graph.k();

    if (subgraph_k > 1) {
      DBG << "initial extend_partition_recursive() for block " << b << ", final k " << final_kb
          << ", subgraph k " << subgraph_k << ", weight " << p_graph.block_weight(b) << " /// "
          << subgraph.total_node_weight();

      extend_partition_recursive(
          subgraph,
          subgraph_partitions[b],
          0,
          subgraph_k,
          final_kb,
          input_ctx,
          subgraph_memory,
          positions[b],
          tmp_extraction_mem_pool_ets,
          bipartitioner_pool,
          &timing
      );
    }
  });
  STOP_TIMER();
  STOP_HEAP_PROFILER();

  START_HEAP_PROFILER("Copy subgraph partitions");
  TIMED_SCOPE("Copy subgraph partitions") {
    p_graph = graph::copy_subgraph_partitions(
        std::move(p_graph), subgraph_partitions, k_prime, input_ctx.partition.k, mapping
    );
  };
  STOP_HEAP_PROFILER();

  auto timings = timings_ets.combine([](BipartitionTimingInfo &a, const BipartitionTimingInfo &b) {
    return a += b;
  });

  if (false) {
    LOG << "bipartitioner_init_ms: "
        << static_cast<std::uint64_t>(timings.bipartitioner_init_ms / 1e6);
    LOG << "bipartitioner_ms:      " << static_cast<std::uint64_t>(timings.bipartitioner_ms / 1e6);
    LOG << "  total_ms:            "
        << static_cast<std::uint64_t>(timings.ip_timings.total_ms / 1e6);
    LOG << "  misc_ms:             "
        << static_cast<std::uint64_t>(timings.ip_timings.misc_ms / 1e6);
    LOG << "  coarsening_ms:       "
        << static_cast<std::uint64_t>(timings.ip_timings.coarsening_ms / 1e6);
    LOG << "    misc_ms:           "
        << static_cast<std::uint64_t>(timings.ip_timings.coarsening_misc_ms / 1e6);
    LOG << "    call_ms:           "
        << static_cast<std::uint64_t>(timings.ip_timings.coarsening_call_ms / 1e6);
    LOG << "      alloc_ms:        "
        << static_cast<std::uint64_t>(timings.ip_timings.coarsening.alloc_ms / 1e6);
    LOG << "      contract_ms:     "
        << static_cast<std::uint64_t>(timings.ip_timings.coarsening.contract_ms / 1e6);
    LOG << "      lp_ms:           "
        << static_cast<std::uint64_t>(timings.ip_timings.coarsening.lp_ms / 1e6);
    LOG << "      interleaved1:    "
        << static_cast<std::uint64_t>(timings.ip_timings.coarsening.interleaved1_ms / 1e6);
    LOG << "      interleaved2:    "
        << static_cast<std::uint64_t>(timings.ip_timings.coarsening.interleaved2_ms / 1e6);
    LOG << "  bipartitioning_ms:   "
        << static_cast<std::uint64_t>(timings.ip_timings.bipartitioning_ms / 1e6);
    LOG << "  uncoarsening_ms:     "
        << static_cast<std::uint64_t>(timings.ip_timings.uncoarsening_ms / 1e6);
    LOG << "graph_init_ms:         " << static_cast<std::uint64_t>(timings.graph_init_ms / 1e6);
    LOG << "extract_ms:            " << static_cast<std::uint64_t>(timings.extract_ms / 1e6);
    LOG << "copy_ms:               " << static_cast<std::uint64_t>(timings.copy_ms / 1e6);
    LOG << "misc_ms:               " << static_cast<std::uint64_t>(timings.misc_ms / 1e6);
  }

  update_partition_context(current_p_ctx, p_graph, input_ctx.partition.k);

  KASSERT(p_graph.k() == k_prime);
}

// extend_partition with local memory allocation for subgraphs
void extend_partition(
    PartitionedGraph &p_graph,
    const BlockID k_prime,
    const Context &input_ctx,
    PartitionContext &current_p_ctx,
    TemporarySubgraphMemoryEts &tmp_extraction_mem_pool_ets,
    InitialBipartitionerWorkerPool &bipartitioner_pool,
    const int num_active_threads
) {
  graph::SubgraphMemory memory;

  memory.resize(
      p_graph.n(),
      input_ctx.partition.k,
      p_graph.m(),
      p_graph.graph().is_node_weighted(),
      p_graph.graph().is_edge_weighted()
  );

  extend_partition(
      p_graph,
      k_prime,
      input_ctx,
      current_p_ctx,
      memory,
      tmp_extraction_mem_pool_ets,
      bipartitioner_pool,
      num_active_threads
  );
}

bool coarsen_once(Coarsener *coarsener, const Graph *graph, PartitionContext &current_p_ctx) {
  SCOPED_TIMER("Coarsening");

  const auto shrunk = coarsener->coarsen();
  const auto &c_graph = coarsener->current();

  // @todo always do this?
  if (shrunk) {
    current_p_ctx.setup(c_graph);
  }

  return shrunk;
}
std::size_t
select_best(const ScalableVector<PartitionedGraph> &p_graphs, const PartitionContext &p_ctx) {
  return select_best(p_graphs.begin(), p_graphs.end(), p_ctx);
}
} // namespace kaminpar::shm::partitioning
