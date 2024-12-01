/*******************************************************************************
 * Generic label propagation benchmark for the shared-memory algorithm.
 *
 * @file:   shm_label_propagation_benchmark.cc
 * @author: Daniel Salwasser
 * @date:   13.12.2023
 ******************************************************************************/
// clang-format off
#include <kaminpar-cli/kaminpar_arguments.h>
// clang-format on

#include <tbb/global_control.h>

#include "kaminpar-shm/coarsening/clustering/lp_clusterer.h"
#include "kaminpar-shm/coarsening/max_cluster_weights.h"
#include "kaminpar-shm/context_io.h"
#include "kaminpar-shm/graphutils/permutator.h"

#include "kaminpar-common/console_io.h"
#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/perf.h"
#include "kaminpar-common/random.h"
#include "kaminpar-common/strutils.h"
#include "kaminpar-common/timer.h"

#include "apps/io/shm_io.h"

using namespace kaminpar;
using namespace kaminpar::shm;

int main(int argc, char *argv[]) {
  // Create context
  Context ctx = create_default_context();

  // Parse CLI arguments
  std::string graph_filename;
  io::GraphFileFormat graph_file_format = io::GraphFileFormat::METIS;
  int seed = 0;

  CLI::App app("Shared-memory LP benchmark");
  app.add_option("-G,--graph", graph_filename, "Graph file")->required();
  app.add_option("-f,--graph-file-format", graph_file_format)
      ->transform(CLI::CheckedTransformer(io::get_graph_file_formats()).description(""))
      ->description(R"(Graph file formats:
  - metis
  - parhip)");
  app.add_option("-t,--threads", ctx.parallel.num_threads, "Number of threads");
  app.add_option("-s,--seed", seed, "Seed for random number generation.")->default_val(seed);
  app.add_option("-k,--k", ctx.partition.k, "Number of blocks in the partition.")->required();
  app.add_option(
         "-e,--epsilon",
         ctx.partition.epsilon,
         "Maximum allowed imbalance, e.g. 0.03 for 3%. Must be strictly positive."
  )
      ->check(CLI::NonNegativeNumber)
      ->capture_default_str();
  create_lp_coarsening_options(&app, ctx);
  create_partitioning_rearrangement_options(&app, ctx);
  create_graph_compression_options(&app, ctx);
  CLI11_PARSE(app, argc, argv);

  tbb::global_control gc(tbb::global_control::max_allowed_parallelism, ctx.parallel.num_threads);
  Random::reseed(seed);

  Graph graph =
      io::read(graph_filename, graph_file_format, ctx.node_ordering, ctx.compression.enabled);
  ctx.setup(graph);
  ctx.debug.graph_name = str::extract_basename(graph_filename);

  cio::print_kaminpar_banner();
  cio::print_build_identifier();
  cio::print_build_datatypes<NodeID, EdgeID, NodeWeight, EdgeWeight>();
  cio::print_delimiter("Input Summary", '#');
  print(ctx, std::cout);

  const double original_epsilon = ctx.partition.epsilon;
  if (ctx.node_ordering == NodeOrdering::DEGREE_BUCKETS) {
    if (ctx.compression.enabled) {
      LOG_WARNING << "A compressed graph cannot be rearranged by degree buckets. Disabling "
                     "degree bucket ordering!";
      ctx.node_ordering = NodeOrdering::NATURAL;
    } else if (!graph.sorted()) {
      CSRGraph &csr_graph = graph.csr_graph();
      graph = graph::rearrange_by_degree_buckets(csr_graph);
    }
  }

  if (graph.sorted()) {
    graph::remove_isolated_nodes(graph, ctx.partition);
  }

  LPClustering lp_clustering(ctx.coarsening);
  lp_clustering.set_max_cluster_weight(compute_max_cluster_weight<NodeWeight>(
      ctx.coarsening, ctx.partition, graph.n(), graph.total_node_weight()
  ));
  lp_clustering.set_desired_cluster_count(0);

  GLOBAL_TIMER.reset();
  ENABLE_HEAP_PROFILER();
  perf::start();

  TIMED_SCOPE("Partitioning") {
    SCOPED_TIMER("Label Propagation");
    SCOPED_HEAP_PROFILER("Label Propagation");

    StaticArray<NodeID> clustering(graph.n());
    lp_clustering.compute_clustering(clustering, graph, false);
  };

  const std::string perf_output = perf::stop();
  DISABLE_HEAP_PROFILER();
  STOP_TIMER();

  if (graph.sorted()) {
    graph::integrate_isolated_nodes(graph, original_epsilon, ctx);
  }

  cio::print_delimiter("Result Summary");
  Timer::global().print_human_readable(std::cout);
  LOG;

  heap_profiler::HeapProfiler::global().set_experiment_summary_options();
  PRINT_HEAP_PROFILE(std::cout);

  LOG << perf_output;

  return EXIT_SUCCESS;
}
