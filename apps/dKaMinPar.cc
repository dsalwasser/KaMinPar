/*******************************************************************************
 * Standalone binary for the distributed partitioner.
 *
 * @file:   dkaminpar.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
// clang-format off
#include "kaminpar-cli/dkaminpar_arguments.h"
#include "kaminpar-dist/context_io.h"
#include "kaminpar-dist/dkaminpar.h"
#include "kaminpar-dist/timer.h"
// clang-format on

#include <kagen.h>
#include <mpi.h>

#ifdef KAMINPAR_ENABLE_TBB_MALLOC
#include <tbb/scalable_allocator.h>
#endif // KAMINPAR_ENABLE_TBB_MALLOC

#include "kaminpar-io/dist_io.h"
#include "kaminpar-io/dist_metis_parser.h"
#include "kaminpar-io/dist_parhip_parser.h"
#include "kaminpar-io/dist_skagen.h"

#include "kaminpar-dist/datastructures/distributed_graph.h"

#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/strutils.h"

#include "apps/version.h"

#ifdef KAMINPAR_HAVE_BACKWARD
#include <backward.hpp>
#endif

#if defined(__linux__)
#include <sys/resource.h>
#endif

using namespace kaminpar;
using namespace kaminpar::dist;

namespace {

enum class IOKind {
  KAMINPAR,
  KAGEN,
  STREAMING_KAGEN
};

std::unordered_map<std::string, IOKind> get_io_kinds() {
  return {
      {"kaminpar", IOKind::KAMINPAR},
      {"kagen", IOKind::KAGEN},
      {"skagen", IOKind::STREAMING_KAGEN},
  };
}

struct ApplicationContext {
  bool dump_config = false;
  bool show_version = false;

  int seed = 0;
  int num_threads = 1;

  int repetitions = 0;

  int max_timer_depth = 3;

  bool heap_profiler_detailed = false;
  int heap_profiler_max_depth = 3;
  bool heap_profiler_print_structs = false;
  float heap_profiler_min_struct_size = 10;

  BlockID k = 0;
  double epsilon = 0.03;
  std::vector<BlockWeight> max_block_weights = {};
  std::vector<double> max_block_weight_factors = {};

  int verbosity = 0;
  bool validate = false;

  IOKind io_kind = IOKind::KAMINPAR;
  GraphDistribution io_distribution = GraphDistribution::BALANCED_EDGES;
  kagen::FileFormat io_format = kagen::FileFormat::EXTENSION;
  kagen::GraphDistribution io_kagen_distribution = kagen::GraphDistribution::BALANCE_EDGES;
  std::string io_skagen_graph_options;
  int io_skagen_chunks_per_pe;

  std::string graph_filename = "";
  std::string partition_filename = "";

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
      ->description(
          R"(Use configuration preset:
  - default, fast: fastest, but lower quality
  - strong:        slower, but higher quality
  - xterapart:     same as default, but use graph compression to reduce peak memory consumption)"
      );

  // Mandatory
  auto *mandatory = cli.add_option_group("Application")->require_option(1);

  // Mandatory -> either dump config ...
  mandatory->add_flag("--dump-config", app.dump_config)
      ->configurable(false)
      ->description(R"(Print the current configuration and exit.
The output should be stored in a file and can be used by the -C,--config option.)");
  mandatory->add_flag("--version", app.show_version, "Show version and exit.");

  // Mandatory -> ... or partition a graph
  auto *gp_group = mandatory->add_option_group("Partitioning")->silent();
  gp_group
      ->add_option(
          "graph,-G,--graph",
          app.graph_filename,
          "Input graph in METIS (file extension *.graph or *.metis) "
          "or binary format (file extension *.bgf)."
      )
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

  cli.add_option("-s,--seed", app.seed, "Seed for random number generation.")
      ->default_val(app.seed);

  cli.add_flag_function("-q,--quiet", [&](auto) { app.verbosity = -1; }, "Suppress all output.");
  cli.add_flag_function(
      "-v,--verbose",
      [&](const auto count) { app.verbosity = count; },
      "Increase output verbosity; can be specified multiple times."
  );

  cli.add_option("-t,--threads", app.num_threads, "Number of threads to be used.")
      ->check(CLI::NonNegativeNumber)
      ->default_val(app.num_threads);

  cli.add_option(
      "--max-timer-depth", app.max_timer_depth, "Set maximum timer depth shown in result summary."
  );
  cli.add_flag_function("-T,--all-timers", [&](auto) {
    app.max_timer_depth = std::numeric_limits<int>::max();
  });

  cli.add_option("--io-format", app.io_format)
      ->transform(CLI::CheckedTransformer(kagen::GetInputFormatMap()).description(""))
      ->description(
          R"(Graph input format. By default, guess the file format from the file extension. Explicit options are:
  - metis:  text format used by the Metis family
  - parhip: binary format used by ParHiP (+ extensions))"
      )
      ->capture_default_str();
  cli.add_option("--io-kind", app.io_kind)
      ->transform(CLI::CheckedTransformer(get_io_kinds()).description(""))
      ->description(R"(Used IO for reading the input graph, possible options are:
  - kaminpar: use KaMinPar for IO
  - kagen:    use KaGen for IO
  - skagen:   use streaming KaGen for IO)")
      ->capture_default_str();
  cli.add_option("--io-distribution", app.io_distribution)
      ->transform(CLI::CheckedTransformer(get_graph_distributions()).description(""))
      ->description(
          R"(Graph distribution scheme used for KaMinPar IO, possible options are:
  - balanced-nodes:        distribute nodes such that each PE has roughly the same number of nodes
  - balanced-edges:        distribute edges such that each PE has roughly the same number of edges
  - balanced-memory-space: distribute graph such that each PE uses roughly the same memory space for the input graph)"
      )
      ->capture_default_str();
  cli.add_option("--io-kagen-distribution", app.io_kagen_distribution)
      ->transform(CLI::CheckedTransformer(kagen::GetGraphDistributionMap()).description(""))
      ->description(R"(Graph distribution scheme used for KaGen IO, possible options are:
  - balance-vertices: distribute vertices such that each PE has roughly the same number of vertices
  - balance-edges:    distribute edges such that each PE has roughly the same number of edges)")
      ->capture_default_str();
  cli.add_option("--io-skagen-graph", app.io_skagen_graph_options)
      ->description("The options used for generating the graph");
  cli.add_option("--io-skagen-chunks", app.io_skagen_chunks_per_pe)
      ->description("The number of chunks per PE that generation will be split into");

  // Heap profiler options
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
  cli.add_flag("--validate", app.validate, "Check input graph for errors.");

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

  cli.add_option(
         "--repetitions",
         app.repetitions,
         "Partition multiple time and output the best cut. Set 0 for just one run."
  )
      ->check(CLI::NonNegativeNumber)
      ->capture_default_str();

  // Algorithmic options
  create_all_options(&cli, ctx);
}

template <typename Lambda> void root_run(Lambda &&l) {
  if (mpi::get_comm_rank(MPI_COMM_WORLD) == 0) {
    l();
  }
}

template <typename Lambda> [[noreturn]] void root_run_and_exit(Lambda &&l) {
  root_run(std::forward<Lambda>(l));
  std::exit(MPI_Finalize());
}

NodeID load_kagen_graph(const ApplicationContext &app, dKaMinPar &partitioner) {
  using namespace kagen;

  START_TIMER("Load KaGen graph");
  KaGen generator(MPI_COMM_WORLD);
  generator.UseCSRRepresentation();
  if (app.validate) {
    generator.EnableUndirectedGraphVerification();
  }
  if (app.verbosity > 1) {
    generator.EnableBasicStatistics();
    generator.EnableOutput(true);
  }

  Graph graph = [&] {
    if (std::find(app.graph_filename.begin(), app.graph_filename.end(), ';') !=
        app.graph_filename.end()) {
      return generator.GenerateFromOptionString(app.graph_filename);
    } else {
      return generator.ReadFromFile(app.graph_filename, app.io_format, app.io_kagen_distribution);
    }
  }();

  KASSERT(graph.vertex_range.second >= graph.vertex_range.first, "invalid vertex range from KaGen");

  auto vtxdist = BuildVertexDistribution<GlobalNodeID>(graph, MPI_UINT64_T, MPI_COMM_WORLD);

  // If data types mismatch, we would need to allocate new memory for the graph; this is to do until
  // we actually need it ...
  std::vector<SInt> xadj = graph.TakeXadj<>();
  std::vector<SInt> adjncy = graph.TakeAdjncy<>();
  std::vector<SSInt> vwgt = graph.TakeVertexWeights<>();
  std::vector<SSInt> adjwgt = graph.TakeEdgeWeights<>();

  static_assert(sizeof(SInt) == sizeof(GlobalNodeID));
  static_assert(sizeof(SInt) == sizeof(GlobalEdgeID));
  static_assert(sizeof(SSInt) == sizeof(GlobalNodeWeight));
  static_assert(sizeof(SSInt) == sizeof(GlobalEdgeWeight));
  STOP_TIMER();

  // Pass the graph to the partitioner --
  partitioner.copy_graph(
      vtxdist,
      {reinterpret_cast<GlobalEdgeID *>(xadj.data()), xadj.size()},
      {reinterpret_cast<GlobalNodeID *>(adjncy.data()), adjncy.size()},
      {reinterpret_cast<GlobalNodeWeight *>(vwgt.data()), vwgt.size()},
      {reinterpret_cast<GlobalEdgeWeight *>(adjwgt.data()), adjwgt.size()}
  );

  return graph.vertex_range.second - graph.vertex_range.first;
}

NodeID load_skagen_graph(const ApplicationContext &app, bool compression, dKaMinPar &partitioner) {
  DistributedGraph graph = [&] {
    if (compression) {
      return DistributedGraph(
          std::make_unique<DistributedCompressedGraph>(io::skagen::compressed_streaming_generate(
              app.io_skagen_graph_options, app.io_skagen_chunks_per_pe, MPI_COMM_WORLD
          ))
      );
    } else {
      return DistributedGraph(
          std::make_unique<DistributedCSRGraph>(io::skagen::csr_streaming_generate(
              app.io_skagen_graph_options, app.io_skagen_chunks_per_pe, MPI_COMM_WORLD
          ))
      );
    }
  }();
  const NodeID n = graph.n();

  partitioner.set_graph(std::move(graph));
  return n;
}

namespace {

kagen::FileFormat map_kagen_format(const kagen::FileFormat format, const std::string &filename) {
  if (format == kagen::FileFormat::EXTENSION) {
    if (str::ends_with(filename, "metis") || str::ends_with(filename, "parhip")) {
      return kagen::FileFormat::METIS;
    } else if (str::ends_with(filename, "parhip") || str::ends_with(filename, "bgf")) {
      return kagen::FileFormat::PARHIP;
    }
  }

  return format;
}

} // namespace

NodeID load_csr_graph(const ApplicationContext &app, dKaMinPar &partitioner) {
  START_TIMER("Load uncompressed graph");
  const auto read_graph = [&] {
    switch (map_kagen_format(app.io_format, app.graph_filename)) {
    case kagen::FileFormat::METIS:
      return io::metis::csr_read(app.graph_filename, app.io_distribution, false, MPI_COMM_WORLD);
    case kagen::FileFormat::PARHIP:
      return io::parhip::csr_read(app.graph_filename, app.io_distribution, false, MPI_COMM_WORLD);
    default:
      root_run_and_exit([&] {
        LOG_ERROR << "To read graphs not stored in METIS/ParHIP format, use KaGen as the IO!";
      });
    }
  };

  DistributedGraph graph(std::make_unique<DistributedCSRGraph>(read_graph()));
  const NodeID n = graph.n();
  STOP_TIMER();

  partitioner.set_graph(std::move(graph));
  return n;
}

NodeID load_compressed_graph(const ApplicationContext &app, dKaMinPar &partitioner) {
  START_TIMER("Load compressed graph");
  const auto read_graph = [&] {
    switch (map_kagen_format(app.io_format, app.graph_filename)) {
    case kagen::FileFormat::METIS:
      return io::metis::compress_read(
          app.graph_filename, app.io_distribution, false, MPI_COMM_WORLD
      );
    case kagen::FileFormat::PARHIP:
      return io::parhip::compressed_read(
          app.graph_filename, app.io_distribution, false, MPI_COMM_WORLD
      );
    default:
      root_run_and_exit([&] {
        LOG_ERROR << "Only graphs stored in METIS or ParHIP format can be compressed!";
      });
    }
  };

  DistributedGraph graph(std::make_unique<DistributedCompressedGraph>(read_graph()));
  const NodeID n = graph.n();
  STOP_TIMER();

  partitioner.set_graph(std::move(graph));
  return n;
}

void report_max_rss() {
  LOG;

#if defined(__linux__)
  if (struct rusage usage; getrusage(RUSAGE_SELF, &usage) == 0) {
    const long rss = usage.ru_maxrss;
    long total_rss, min_rss, max_rss;

    MPI_Reduce(&rss, &total_rss, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&rss, &min_rss, 1, MPI_LONG, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&rss, &max_rss, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    LOG << "Maximum resident set size: " << total_rss << " kB [min: " << min_rss
        << " kB, avg: " << total_rss / size << " kB, max: " << max_rss << " kB]";
  } else {
#else
  {
#endif
    LOG << "Maximum resident set size: unknown\n";
  }
}

GlobalEdgeWeight run_partitioner_once(
    dKaMinPar &partitioner, std::vector<BlockID> &partition, const ApplicationContext &app
) {
  partitioner.set_k(app.k);

  if (!app.max_block_weight_factors.empty()) {
    const double total_factor = std::accumulate(
        app.max_block_weight_factors.begin(), app.max_block_weight_factors.end(), 0.0
    );

    if (total_factor <= 1.0) {
      LOG_ERROR << "Error: total block weights must be greater than the total node weight; "
                << "this is not the case with the given factors.";
      std::exit(1);
    }

    partitioner.set_relative_max_block_weights(std::move(app.max_block_weight_factors));

  } else if (!app.max_block_weights.empty()) {
    const BlockWeight total_block_weight = std::accumulate(
        app.max_block_weights.begin(), app.max_block_weights.end(), static_cast<BlockWeight>(0)
    );
    const NodeWeight total_node_weight = partitioner.graph()->global_total_node_weight();

    if (total_node_weight >= total_block_weight) {
      LOG_ERROR << "Error: total max block weights (" << total_block_weight
                << ") must be greater than the total node weight (" << total_node_weight << ").";
      std::exit(1);
    }

    partitioner.set_absolute_max_block_weights(std::move(app.max_block_weights));
  } else {
    partitioner.set_uniform_max_block_weights(app.epsilon);
  }

  return partitioner.compute_partition(partition);
}

void run_partitioner(
    dKaMinPar &partitioner, std::vector<BlockID> &partition, const ApplicationContext &app
) {
#ifdef KAMINPAR_HAVE_BACKWARD
  backward::MPIErrorHandler mpi_error_handler(MPI_COMM_WORLD);
  backward::SignalHandling sh;
#endif // KAMINPAR_HAVE_BACKWARD

  if (app.repetitions == 0) {
    run_partitioner_once(partitioner, partition, app);
    if (app.verbosity >= 0) {
      report_max_rss();
    }
    return;
  }

  START_HEAP_PROFILER("Input Graph Allocation");
  std::vector<GlobalEdgeWeight> cuts;
  GlobalEdgeWeight best_cut = std::numeric_limits<GlobalEdgeWeight>::max();
  std::vector<BlockID> best_partition(partition.size());
  STOP_HEAP_PROFILER();

  for (int repetition = 0; repetition < app.repetitions; ++repetition) {
    const GlobalEdgeWeight cut = run_partitioner_once(partitioner, partition, app);
    cuts.push_back(cut);

    if (cut < best_cut) {
      std::swap(best_partition, partition);
      best_cut = cut;
    }

    if (app.verbosity >= 0) {
      report_max_rss();
    }
  }

  root_run([&] {
    const GlobalEdgeWeight cut_sum =
        std::accumulate(cuts.begin(), cuts.end(), static_cast<GlobalNodeWeight>(0));
    const double avg_cut = 1.0 * cut_sum / cuts.size();
    const GlobalEdgeWeight worst_cut = *std::max_element(cuts.begin(), cuts.end());

    LOG;
    LOG << "Worst cut: " << worst_cut;
    LOG << "Avg. cut:  " << avg_cut;
    LOG << "Best cut:  " << best_cut;
  });
}

} // namespace

int main(int argc, char *argv[]) {
  int provided, rank;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  CLI::App cli(
      "dKaMinPar: (Somewhat) Minimal Distributed Deep Multilevel "
      "Graph Partitioner"
  );
  ApplicationContext app;
  Context ctx = create_default_context();
  setup_context(cli, app, ctx);
  CLI11_PARSE(cli, argc, argv);

  if (app.dump_config) {
    root_run_and_exit([&] {
      CLI::App dump;
      create_all_options(&dump, ctx);
      std::cout << dump.config_to_str(true, true);
    });
  }

  if (app.show_version) {
    root_run_and_exit([&] { print_version(); });
  }

  if (app.dry_run) {
    std::exit(MPI_Finalize());
  }

#ifdef KAMINPAR_ENABLE_TBB_MALLOC
  // If available, use huge pages for large allocations
  scalable_allocation_mode(TBBMALLOC_USE_HUGE_PAGES, !app.no_huge_pages);
#endif // KAMINPAR_ENABLE_TBB_MALLOC

  ENABLE_HEAP_PROFILER();

  dKaMinPar partitioner(MPI_COMM_WORLD, app.num_threads, ctx);
  dKaMinPar::reseed(app.seed);

  if (app.verbosity < 0) {
    partitioner.set_output_level(OutputLevel::QUIET);
  } else if (app.verbosity == 1) {
    partitioner.set_output_level(OutputLevel::EXPERIMENT);
  } else if (app.verbosity >= 2) {
    partitioner.set_output_level(OutputLevel::DEBUG);
  }

  partitioner.context().debug.graph_filename = str::extract_basename(app.graph_filename);
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

  START_HEAP_PROFILER("Input Graph Allocation");
  // Load the graph via KaGen or via our graph compressor.
  const NodeID n = [&] {
    SCOPED_TIMER("IO");

    if (app.io_kind == IOKind::KAMINPAR) {
      if (ctx.compression.enabled) {
        return load_compressed_graph(app, partitioner);
      }

      return load_csr_graph(app, partitioner);
    }

    if (app.io_kind == IOKind::STREAMING_KAGEN) {
      return load_skagen_graph(app, ctx.compression.enabled, partitioner);
    } else if (ctx.compression.enabled) {
      root_run([] {
        LOG_WARNING << "Disabling graph compression since it is not supported with KaGen-IO!";
      });
      ctx.compression.enabled = false;
    }

    return load_kagen_graph(app, partitioner);
  }();

  // Allocate memory for the partition
  std::vector<BlockID> partition(n);
  STOP_HEAP_PROFILER();

  run_partitioner(partitioner, partition, app);

  if (!app.partition_filename.empty()) {
    dist::io::partition::write(app.partition_filename, partition);
  }

  DISABLE_HEAP_PROFILER();

  return MPI_Finalize();
}
