/*******************************************************************************
 * Standalone binary for the shared-memory partitioner.
 *
 * @file:   KaMinPar.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
// clang-format off
#include "kaminpar-cli/kaminpar_arguments.h"
// clang-format on

#include <cstdlib>
#include <iostream>
#include <span>

#ifdef KAMINPAR_ENABLE_TBB_MALLOC
#include <tbb/scalable_allocator.h>
#endif // KAMINPAR_ENABLE_TBB_MALLOC

#if __has_include(<numa.h>)
#include <numa.h>
#endif // __has_include(<numa.h>)

#include "kaminpar-io/kaminpar_io.h"

#include "kaminpar-shm/graphutils/graph_validator.h"
#include "kaminpar-shm/graphutils/permutator.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/strutils.h"
#include "kaminpar-common/timer.h"

#include "apps/version.h"

#if defined(__linux__)
#include <sys/resource.h>
#endif

using namespace kaminpar;
using namespace kaminpar::shm;

namespace {

struct ApplicationContext {
  bool dump_config = false;
  bool show_version = false;

  int seed = 0;
  int num_threads = tbb::this_task_arena::max_concurrency();

  int max_timer_depth = 3;

  bool heap_profiler_detailed = false;
  int heap_profiler_max_depth = 3;
  bool heap_profiler_print_structs = false;
  float heap_profiler_min_struct_size = 10;

  BlockID k = 0;
  double epsilon = 0.03;
  std::vector<BlockWeight> max_block_weights = {};
  std::vector<double> max_block_weight_factors = {};

  double min_epsilon = 0.0;
  std::vector<BlockWeight> min_block_weights = {};
  std::vector<double> min_block_weight_factors = {};
  bool no_empty_blocks = false;

  int verbosity = 0;
  bool validate = false;

  std::string graph_filename = "";
  io::GraphFileFormat input_graph_file_format = io::GraphFileFormat::METIS;

  bool ignore_node_weights = false;
  bool ignore_edge_weights = false;

  std::string partition_filename = "";
  std::string rearranged_graph_filename = "";
  std::string rearranged_mapping_filename = "";
  std::string block_sizes_filename = "";
  io::GraphFileFormat output_graph_file_format = io::GraphFileFormat::METIS;

  bool no_huge_pages = false;

  bool dry_run = false;
};

void setup_context(CLI::App &cli, ApplicationContext &app, Context &ctx) {
  cli.set_config("-C,--config", "", "Read parameters from a TOML configuration file.", false);
  cli.add_option_function<std::string>(
         "-P,--preset",
         [&](const std::string preset) { ctx = create_context_by_preset_name(preset); }
  )
      ->check(CLI::IsMember(get_preset_names()))
      ->description(R"(Use configuration preset:
  - fast:     fastest (especially for small graphs), but lowest quality
  - default:  in-between
  - terapart: same as default, but use graph compression to reduce peak memory consumption
  - strong:   slower, but higher quality (LP + FM)
  - largek:   tuned for k > 1024-ish)");

  // Mandatory
  auto *mandatory = cli.add_option_group("Application")->require_option(1);

  // Mandatory -> either dump config ...
  mandatory->add_flag("--dump-config", app.dump_config)
      ->configurable(false)
      ->description(R"(Print the current configuration and exit.
The output should be stored in a file and can be used by the -C,--config option.)");
  mandatory->add_flag("--version", app.show_version, "Show version and exit.");

  // Mandatory -> ... or partition a graph
  auto *gp_group = mandatory->add_option_group("Partitioning");
  gp_group->add_option("graph,-G,--graph", app.graph_filename, "Input graph in METIS format.")
      ->check(CLI::ExistingFile)
      ->configurable(false);

  auto *partition_group = gp_group->add_option_group("Partition settings")->require_option(1);
  partition_group
      ->add_option(
          "k,-k,--k",
          app.k,
          "Number of blocks in the partition. This option will be ignored if explicit block "
          "weights are specified via --block-weights or --block-weight-factors."
      )
      ->check(CLI::Range(static_cast<BlockID>(2), std::numeric_limits<BlockID>::max()));
  partition_group
      ->add_option(
          "-B,--block-weights",
          app.max_block_weights,
          "Absolute max block weights, one weight for each block of the partition. If this "
          "option is set, --epsilon will be ignored."
      )
      ->check(CLI::NonNegativeNumber)
      ->capture_default_str();
  partition_group->add_option(
      "-b,--block-weight-factors",
      app.max_block_weight_factors,
      "Max block weights relative to the total node weight of the input graph, one factor for each "
      "block of the partition. If this option is set, --epsilon will be ignored."
  );

  // Application options
  cli.add_option(
         "-e,--epsilon",
         app.epsilon,
         "Maximum allowed imbalance, e.g. 0.03 for 3%. Must be greater than 0%. If maximum block "
         "weights are specified explicitly via the --block-weights, this option will be ignored."
  )
      ->check(CLI::NonNegativeNumber)
      ->capture_default_str();

  cli.add_option(
         "--min-epsilon",
         app.min_epsilon,
         "Maximum allowed imbalance for minimum block weights, e.g., 0.03 for 3%. Minimum block "
         "weight imbalance is ignored when set to 0% (default)."
  )
      ->check(CLI::NonNegativeNumber)
      ->capture_default_str();

  cli.add_option(
         "--min-block-weights",
         app.min_block_weights,
         "Absolute minimum block weights, one weight for each block of the partition. If this "
         "option is set, --min-epsilon will be ignored."
  )
      ->check(CLI::NonNegativeNumber)
      ->capture_default_str();

  cli.add_option(
         "--min-block-weight-factors",
         app.min_block_weight_factors,
         "Min block weights relative to the total node weight of the input graph, one factor for "
         "each block of the partition. If this option is set, --min-epsilon will be ignored."
  )
      ->check(CLI::NonNegativeNumber)
      ->capture_default_str();

  cli.add_flag("--no-empty-blocks", app.no_empty_blocks, "Forbid empty blocks.");

  cli.add_option("-s,--seed", app.seed, "Seed for random number generation.")
      ->default_val(app.seed);

  cli.add_flag_function("-q,--quiet", [&](auto) { app.verbosity = -1; }, "Suppress all output.");
  cli.add_flag_function(
      "-v,--verbose",
      [&](const auto count) { app.verbosity = count; },
      "Increase output verbosity; can be specified multiple times."
  );

  cli.add_option("-t,--threads", app.num_threads, "Number of threads to be used.")
      ->check(CLI::PositiveNumber)
      ->default_val(app.num_threads);

  cli.add_option(
      "--max-timer-depth", app.max_timer_depth, "Set maximum timer depth shown in result summary."
  );
  cli.add_flag_function("-T,--all-timers", [&](auto) {
    app.max_timer_depth = std::numeric_limits<int>::max();
  });
  cli.add_option("-f,--graph-file-format,--input-graph-file-format", app.input_graph_file_format)
      ->transform(
          CLI::CheckedTransformer(
              std::unordered_map<std::string, io::GraphFileFormat>{
                  {"metis", io::GraphFileFormat::METIS},
                  {"parhip", io::GraphFileFormat::PARHIP},
                  {"compressed", io::GraphFileFormat::COMPRESSED},
              },
              CLI::ignore_case
          )
      )
      ->description(R"(Graph file formats:
  - metis
  - parhip
  - compressed)")
      ->capture_default_str();

  cli.add_flag(
      "--ignore-node-weights",
      app.ignore_node_weights,
      "Ignore the node weights of the input graph (replace with unit weights)."
  );
  cli.add_flag(
      "--ignore-edge-weights",
      app.ignore_edge_weights,
      "Ignore the edge weights of the input graph (replace with unit weights)."
  );

  if constexpr (kHeapProfiling) {
    auto *hp_group = cli.add_option_group("Heap Profiler");

    hp_group
        ->add_flag(
            "-H,--hp-print-detailed",
            app.heap_profiler_detailed,
            "Show all levels in the result summary."
        )
        ->capture_default_str();
    hp_group
        ->add_option(
            "--hp-max-depth",
            app.heap_profiler_max_depth,
            "Set maximum heap profiler depth shown in the result summary."
        )
        ->capture_default_str();
    hp_group
        ->add_flag(
            "--hp-print-structs",
            app.heap_profiler_print_structs,
            "Print data structure memory statistics in the result summary."
        )
        ->capture_default_str();
    hp_group
        ->add_option(
            "--hp-min-struct-size",
            app.heap_profiler_min_struct_size,
            "Sets the minimum size of a data structure in MiB to be included in the result summary."
        )
        ->capture_default_str()
        ->check(CLI::NonNegativeNumber);
  }

  cli.add_option("-o,--output", app.partition_filename, "Output filename for the graph partition.")
      ->capture_default_str();
  cli.add_option(
         "--output-rearranged-graph",
         app.rearranged_graph_filename,
         "Output filename for the rearranged graph: rearranged input graph such that the vertices "
         "of each block form a consecutive range. The corresponding mapping can be saved using the "
         "--output-rearranged-graph-mapping option."
  )
      ->capture_default_str();
  cli.add_option(
         "--output-rearranged-graph-mapping",
         app.rearranged_mapping_filename,
         "Output filename for the mapping corresponding to the rearranged input graph (see "
         "--output-rerranged-graph, only works in combination with this option)."
  )
      ->capture_default_str();
  cli.add_option("--output-graph-file-format", app.output_graph_file_format)
      ->transform(
          CLI::CheckedTransformer(
              std::unordered_map<std::string, io::GraphFileFormat>{
                  {"metis", io::GraphFileFormat::METIS},
                  {"parhip", io::GraphFileFormat::PARHIP},
                  {"compressed", io::GraphFileFormat::COMPRESSED},
              },
              CLI::ignore_case
          )
      )
      ->description(R"(Graph file formats:
  - metis
  - parhip
  - compressed)")
      ->capture_default_str();
  cli.add_option(
         "--output-block-sizes",
         app.block_sizes_filename,
         "Output the number of vertices in each block (one line per block)."
  )
      ->capture_default_str();

  cli.add_flag(
      "--validate",
      app.validate,
      "Validate input parameters before partitioning (currently only "
      "checks the graph format)."
  );
  cli.add_flag("--no-huge-pages", app.no_huge_pages, "Do not use huge pages via TBBmalloc.");

  cli.add_option(
         "--max-overcommitment-factor",
         heap_profiler::max_overcommitment_factor,
         "Limit memory overcommitment to this factor times the total available system memory."
  )
      ->capture_default_str();
  cli.add_flag(
         "--bruteforce-max-overcommitment-factor",
         heap_profiler::bruteforce_max_overcommitment_factor,
         "If enabled, the maximum overcommitment factor is slowly decreased until memory "
         "overcommitment succeeded."
  )
      ->capture_default_str();

  cli.add_flag(
      "--dry-run",
      app.dry_run,
      "Only check the given command line arguments, but do not partition the graph."
  );

  // Algorithmic options
  create_all_options(&cli, ctx);
}

inline void
output_rearranged_graph(const ApplicationContext &app, const std::vector<BlockID> &partition) {
  if (app.rearranged_graph_filename.empty()) {
    return;
  }

  auto graph = io::read_graph(app.graph_filename, app.input_graph_file_format);
  if (!graph) {
    LOG_ERROR << "Could not output rearranged graph as the input graph cannot be read.";
    return;
  }

  auto &csr_graph = graph.value().csr_graph();
  auto permutations = shm::graph::compute_node_permutation_by_generic_buckets(
      csr_graph.n(), app.k, [&](const NodeID u) { return partition[u]; }
  );

  if (!app.rearranged_mapping_filename.empty()) {
    io::write_remapping(app.rearranged_mapping_filename, permutations.old_to_new);
  }

  StaticArray<EdgeID> tmp_nodes(csr_graph.raw_nodes().size());
  StaticArray<NodeID> tmp_edges(csr_graph.raw_edges().size());
  StaticArray<NodeWeight> tmp_node_weights(csr_graph.raw_node_weights().size());
  StaticArray<EdgeWeight> tmp_edge_weights(csr_graph.raw_edge_weights().size());

  shm::graph::build_permuted_graph(
      csr_graph.raw_nodes(),
      csr_graph.raw_edges(),
      csr_graph.raw_node_weights(),
      csr_graph.raw_edge_weights(),
      permutations,
      tmp_nodes,
      tmp_edges,
      tmp_node_weights,
      tmp_edge_weights
  );

  Graph permuted_graph = {std::make_unique<CSRGraph>(
      std::move(tmp_nodes),
      std::move(tmp_edges),
      std::move(tmp_node_weights),
      std::move(tmp_edge_weights)
  )};

  io::write_graph(app.rearranged_graph_filename, app.output_graph_file_format, permuted_graph);
}

inline void print_rss(const ApplicationContext &app) {
  if (app.verbosity >= 0) {
    std::cout << "\n";

#if defined(__linux__)
    if (struct rusage usage; getrusage(RUSAGE_SELF, &usage) == 0) {
      std::cout << "Maximum resident set size: " << usage.ru_maxrss << " KiB\n";
    } else {
#else
    {
#endif
      std::cout << "Maximum resident set size: unknown\n";
    }

    std::cout << std::flush;
  }
}

} // namespace

int main(int argc, char *argv[]) {
#if __has_include(<numa.h>)
  if (numa_available() >= 0) {
    numa_set_interleave_mask(numa_all_nodes_ptr);
  }
#endif // __has_include(<numa.h>)

  CLI::App cli("KaMinPar: (Somewhat) Minimal Deep Multilevel Graph Partitioner");
  ApplicationContext app;
  Context ctx = create_default_context();
  setup_context(cli, app, ctx);
  CLI11_PARSE(cli, argc, argv);

  if (app.dump_config) {
    CLI::App dump;
    create_all_options(&dump, ctx);
    std::cout << dump.config_to_str(true, true);
    std::exit(1);
  }

  if (app.show_version) {
    print_version();
    std::exit(0);
  }

  if (app.dry_run) {
    std::exit(0);
  }

#ifdef KAMINPAR_ENABLE_TBB_MALLOC
  // If available, use huge pages for large allocations
  scalable_allocation_mode(TBBMALLOC_USE_HUGE_PAGES, !app.no_huge_pages);
#endif // KAMINPAR_ENABLE_TBB_MALLOC

  ENABLE_HEAP_PROFILER();

  // Setup the KaMinPar instance
  KaMinPar partitioner(app.num_threads, ctx);
  KaMinPar::reseed(app.seed);

  if (app.verbosity < 0) {
    partitioner.set_output_level(OutputLevel::QUIET);
  } else if (app.verbosity == 1) {
    partitioner.set_output_level(OutputLevel::EXPERIMENT);
  } else if (app.verbosity >= 2) {
    partitioner.set_output_level(OutputLevel::DEBUG);
  }

  partitioner.context().debug.graph_name = str::extract_basename(app.graph_filename);
  partitioner.set_max_timer_depth(app.max_timer_depth);

  if constexpr (kHeapProfiling) {
    auto &global_heap_profiler = heap_profiler::HeapProfiler::global();

    global_heap_profiler.set_max_depth(app.heap_profiler_max_depth);
    global_heap_profiler.set_print_data_structs(app.heap_profiler_print_structs);
    global_heap_profiler.set_min_data_struct_size(app.heap_profiler_min_struct_size);

    if (app.heap_profiler_detailed) {
      global_heap_profiler.set_experiment_summary_options();
    }
  }

  // Read the input graph and allocate memory for the partition
  START_HEAP_PROFILER("Input Graph Allocation");
  Graph graph = TIMED_SCOPE("Read input graph") {
    if (auto graph = io::read_graph(
            app.graph_filename,
            app.input_graph_file_format,
            ctx.compression.enabled,
            ctx.node_ordering
        )) {
      return std::move(*graph);
    }

    LOG_ERROR << "Failed to read the input graph.";
    std::exit(EXIT_FAILURE);
  };

  if (app.ignore_node_weights && !ctx.compression.enabled) {
    auto &csr_graph = graph.csr_graph();
    graph = Graph(
        std::make_unique<CSRGraph>(
            csr_graph.take_raw_nodes(),
            csr_graph.take_raw_edges(),
            StaticArray<NodeWeight>(),
            csr_graph.take_raw_edge_weights()
        )
    );
  } else if (app.ignore_node_weights) {
    LOG_WARNING << "Cannot ignore node weights: only supported for uncompressed graphs.";
  }

  if (app.ignore_edge_weights && !ctx.compression.enabled) {
    auto &csr_graph = graph.csr_graph();
    graph = Graph(
        std::make_unique<CSRGraph>(
            csr_graph.take_raw_nodes(),
            csr_graph.take_raw_edges(),
            csr_graph.take_raw_node_weights(),
            StaticArray<EdgeWeight>()
        )
    );
  } else if (app.ignore_edge_weights) {
    LOG_WARNING << "Cannot ignore edge weights: only supported for uncompressed graphs.";
  }

  if (app.validate && !ctx.compression.enabled) {
    shm::validate_undirected_graph(graph);
  } else if (app.validate) {
    LOG_WARNING << "Cannot validate the input graph: only supported for uncompressed graphs.";
  }

  if (static_cast<std::uint64_t>(graph.m()) >
      static_cast<std::uint64_t>(std::numeric_limits<EdgeWeight>::max())) {
    LOG_WARNING << "The edge weight type is not large enough to store the sum of all edge weights. "
                << "This might cause overflows for very large cuts.";
  }

  RECORD("partition") std::vector<BlockID> partition(graph.n());
  RECORD_LOCAL_DATA_STRUCT(partition, partition.capacity() * sizeof(BlockID));
  STOP_HEAP_PROFILER();

  // Compute partition
  partitioner.set_graph(std::move(graph));
  partitioner.set_k(app.k);

  if (!app.min_block_weight_factors.empty()) {
    const double total_factor = std::accumulate(
        app.min_block_weight_factors.begin(), app.min_block_weight_factors.end(), 0.0
    );

    if (total_factor >= 1.0) {
      LOG_ERROR << "Error: total min block weights must be smaller than the total node weight; "
                << "this is not the case with the given factors.";
      std::exit(1);
    }

    partitioner.set_relative_min_block_weights(app.min_block_weight_factors);
  } else if (!app.min_block_weights.empty()) {
    const BlockWeight total_block_weight = std::accumulate(
        app.min_block_weights.begin(), app.min_block_weights.end(), static_cast<BlockWeight>(0)
    );

    const NodeWeight total_node_weight = partitioner.graph()->total_node_weight();

    if (total_node_weight <= total_block_weight) {
      LOG_ERROR << "Error: total min block weights (" << total_block_weight
                << ") must be smaller than the total node weight (" << total_node_weight << ").";
      std::exit(1);
    }

    partitioner.set_absolute_min_block_weights(app.min_block_weights);
  } else if (app.min_epsilon > 0.0) {
    partitioner.set_uniform_min_block_weights(app.min_epsilon);
  } else if (app.no_empty_blocks) {
    partitioner.set_absolute_min_block_weights(std::vector<BlockWeight>(app.k, 1));
  }

  if (!app.max_block_weight_factors.empty()) {
    const double total_factor = std::accumulate(
        app.max_block_weight_factors.begin(), app.max_block_weight_factors.end(), 0.0
    );

    if (total_factor <= 1.0) {
      LOG_ERROR << "Error: total block weights must be greater than the total node weight; "
                << "this is not the case with the given factors.";
      std::exit(1);
    }

    partitioner.set_relative_max_block_weights(app.max_block_weight_factors);
  } else if (!app.max_block_weights.empty()) {
    const BlockWeight total_block_weight = std::accumulate(
        app.max_block_weights.begin(), app.max_block_weights.end(), static_cast<BlockWeight>(0)
    );

    const NodeWeight total_node_weight = partitioner.graph()->total_node_weight();

    if (total_node_weight >= total_block_weight) {
      LOG_ERROR << "Error: total max block weights (" << total_block_weight
                << ") must be greater than the total node weight (" << total_node_weight << ").";
      std::exit(1);
    }

    partitioner.set_absolute_max_block_weights(app.max_block_weights);
  } else {
    partitioner.set_uniform_max_block_weights(app.epsilon);
  }

  partitioner.compute_partition(partition);

  // Save graph partition
  if (!app.partition_filename.empty()) {
    io::write_partition(app.partition_filename, partition);
  }

  if (!app.block_sizes_filename.empty()) {
    io::write_block_sizes(app.block_sizes_filename, app.k, partition);
  }

  if (!app.rearranged_graph_filename.empty()) {
    output_rearranged_graph(app, partition);
  }

  DISABLE_HEAP_PROFILER();

  print_rss(app);

  return 0;
}
