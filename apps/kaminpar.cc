/*******************************************************************************
 * @file:   kaminpar.cc
 *
 * @author: Daniel Seemaier
 * @date:   21.09.21
 * @brief:  KaMinPar binary. Use --help for information on how to use this
 * program.
 ******************************************************************************/
#include "apps/apps.h"
#include "kaminpar/algorithm/graph_permutation.h"
#include "kaminpar/application/arguments.h"
#include "kaminpar/application/arguments_parser.h"
#include "kaminpar/datastructure/graph.h"
#include "kaminpar/definitions.h"
#include "kaminpar/io.h"
#include "kaminpar/metrics.h"
#include "kaminpar/partitioning_scheme/partitioning.h"
#include "kaminpar/utility/console_io.h"
#include "kaminpar/utility/logger.h"
#include "kaminpar/utility/random.h"
#include "kaminpar/utility/timer.h"

#include <chrono>
#include <iostream>
#include <tbb/parallel_for.h>

using namespace kaminpar;
using namespace std::string_literals;

// clang-format off
void sanitize_context(const Context &context) {
  ALWAYS_ASSERT(!std::ifstream(context.graph_filename) == false)
      << "Graph file cannot be read. Ensure that the file exists and is readable.";
  ALWAYS_ASSERT(!context.save_partition || !std::ofstream(context.partition_file()) == false)
      << "Partition file cannot be written to " << context.partition_file() << "."
      << "Ensure that the directory exists and is writable.";
  ALWAYS_ASSERT(context.partition.k >= 2) << "k must be at least 2.";
  ALWAYS_ASSERT(context.partition.epsilon >= 0) << "Balance constraint cannot be negative.";
  ALWAYS_ASSERT(context.partition.epsilon > 0) << "Epsilon cannot be zero.";

  // Coarsening
  ALWAYS_ASSERT(context.coarsening.contraction_limit >= 2) << "Contraction limit must be at least 2.";

  // Initial Partitioning
  ALWAYS_ASSERT(context.initial_partitioning.max_num_repetitions >= context.initial_partitioning.min_num_repetitions)
      << "Maximum number of repetitions should be at least as large as the minimum number of repetitions.";

  // Initial Partitioning -> Coarsening
  ALWAYS_ASSERT(context.initial_partitioning.coarsening.contraction_limit >= 2);
}
// clang-format on

void print_statistics(const PartitionedGraph &p_graph, const Context &ctx) {
  const EdgeWeight cut = metrics::edge_cut(p_graph);
  const double imbalance = metrics::imbalance(p_graph);
  const bool feasible = metrics::is_feasible(p_graph, ctx.partition);

  // statistics output that is easy to parse
  if (!ctx.quiet) { Timer::global().print_machine_readable(std::cout); }
  LOG << "RESULT cut=" << cut << " imbalance=" << imbalance << " feasible=" << feasible << " k=" << p_graph.k();
  LOG;

  // statistics output that is easy to read
  if (!ctx.quiet) { Timer::global().print_human_readable(std::cout); }
  LOG;
  LOG << "-> k=" << p_graph.k();
  LOG << "-> cut=" << cut;
  LOG << "-> imbalance=" << imbalance;
  LOG << "-> feasible=" << feasible;
  if (p_graph.k() <= 512) {
    LOG << "-> block weights:";
    LOG << logger::TABLE << p_graph.block_weights();
  }
  if (p_graph.k() != ctx.partition.k || !feasible) { LOG_ERROR << "*** Partition is infeasible!"; }
}

std::string generate_partition_filename(const Context &ctx) {
  std::stringstream filename;
  filename << utility::str::extract_basename(ctx.graph_filename);
  filename << "__t" << ctx.parallel.num_threads;
  filename << "__k" << ctx.partition.k;
  filename << "__eps" << ctx.partition.epsilon;
  filename << "__seed" << ctx.seed;
  filename << ".partition";
  return filename.str();
}

int main(int argc, char *argv[]) {
  // keep alive
  const auto sh = init_backward();
  UNUSED(sh); // hide compile warning if backward is non use

  //
  // Parse command line arguments, sanitize, generate output filenames
  //
  Context ctx;
  try {
    ctx = app::parse_options(argc, argv);
    if (ctx.partition_filename.empty()) { ctx.partition_filename = generate_partition_filename(ctx); }
    sanitize_context(ctx);
  } catch (const std::runtime_error &e) { FATAL_ERROR << e.what(); }
  if (ctx.debug.just_sanitize_args) { std::exit(0); }

  if (ctx.partition.fast_initial_partitioning) {
    ctx.initial_partitioning.min_num_repetitions = 4;
    ctx.initial_partitioning.min_num_non_adaptive_repetitions = 2;
    ctx.initial_partitioning.max_num_repetitions = 4;
  }

  Logger::set_quiet_mode(ctx.quiet);

  print_identifier(argc, argv);

  //
  // Initialize
  //
  Randomize::seed = ctx.seed;
  auto gc = init_parallelism(ctx.parallel.num_threads); // must stay alive
  if (ctx.parallel.use_interleaved_numa_allocation) { init_numa(); }

  //
  // Load input graph
  //
  bool remove_isolated_nodes = false;
  const double original_epsilon = ctx.partition.epsilon;

  auto [graph, permutations] = [&] {
    StaticArray<EdgeID> nodes;
    StaticArray<NodeID> edges;
    StaticArray<NodeWeight> node_weights;
    StaticArray<EdgeWeight> edge_weights;

    const io::metis::GraphInfo info = TIMED_SCOPE("IO") {
      return io::metis::read(ctx.graph_filename, nodes, edges, node_weights, edge_weights);
    };

    START_TIMER("Partitioning");
    START_TIMER("Preprocessing");

    // sort nodes by degree bucket and rearrange graph, remove isolated nodes
    remove_isolated_nodes = info.has_isolated_nodes && ctx.partition.remove_isolated_nodes;
    auto permutations = graph::rearrange_and_remove_isolated_nodes(remove_isolated_nodes, ctx.partition, nodes, edges,
                                                                   node_weights, edge_weights,
                                                                   static_cast<NodeWeight>(info.total_node_weight));
    STOP_TIMER();
    STOP_TIMER();

    return std::pair{Graph{std::move(nodes), std::move(edges), std::move(node_weights), std::move(edge_weights), true},
                     std::move(permutations)};
  }();

  //
  // Setup graph dependent context parameters
  //
  ctx.setup(graph);

  cio::print_banner("Input parameters");
  LOG << "CONTEXT " << ctx;
  LOG << "INPUT graph=" << ctx.graph_filename << " "
      << "n=" << graph.n() << " "
      << "m=" << graph.m() << " "
      << "k=" << ctx.partition.k << " "
      << "epsilon=" << ctx.partition.epsilon << " ";
  LOG << "==> max_block_weight=" << ctx.partition.max_block_weight(0);

  //
  // Perform actual partitioning
  //
  PartitionedGraph p_graph = partitioning::partition(graph, ctx);

  //
  // Re-add isolated nodes (if they were removed)
  //
  if (remove_isolated_nodes) {
    cio::print_banner("Postprocessing");

    START_TIMER("Partitioning");
    START_TIMER("Postprocessing");

    const NodeID num_nonisolated_nodes = graph.n(); // this becomes the first isolated node
    graph.raw_nodes().unrestrict();
    graph.raw_node_weights().unrestrict();
    graph.update_total_node_weight();
    const NodeID num_isolated_nodes = graph.n() - num_nonisolated_nodes;

    // note: max block weights should not change
    ctx.setup(graph);
    ctx.partition.epsilon = original_epsilon;
    ctx.partition.setup_max_block_weight();

    LOG << "Add " << num_isolated_nodes << " isolated nodes and revert to epsilon=" << original_epsilon;
    LOG << "==> max_block_weight=" << ctx.partition.max_block_weight(0);
    p_graph = graph::revert_isolated_nodes_removal(std::move(p_graph), num_isolated_nodes, ctx.partition);
    STOP_TIMER();
    STOP_TIMER();
  }

  //
  // Store output partition (if requested)
  //
  if (ctx.save_partition) {
    SCOPED_TIMER("IO");
    io::partition::write(ctx.partition_file(), p_graph, permutations.old_to_new);
    LOG << "Wrote partition to: " << ctx.partition_file();
  }

  //
  // Print some statistics
  //
  STOP_TIMER(); // stop root timer

  cio::print_banner("Statistics");
  print_statistics(p_graph, ctx);
  return 0;
}
