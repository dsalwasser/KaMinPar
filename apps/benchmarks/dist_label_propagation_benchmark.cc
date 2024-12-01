/*******************************************************************************
 * Generic label propagation benchmark for the distributed-memory algorithm.
 *
 * @file:   dist_label_propagation_benchmark.cc
 * @author: Daniel Salwasser
 * @date:   23.11.2024
 ******************************************************************************/
// clang-format off
#include <kaminpar-cli/dkaminpar_arguments.h>
// clang-format on

#include <cstdlib>
#include <string>
#include <unordered_map>

#include <mpi.h>
#include <tbb/global_control.h>

#include "kaminpar-mpi/utils.h"
#include "kaminpar-mpi/wrapper.h"

#include "kaminpar-dist/coarsening/clustering/lp/global_lp_clusterer.h"
#include "kaminpar-dist/context.h"
#include "kaminpar-dist/context_io.h"
#include "kaminpar-dist/datastructures/distributed_compressed_graph.h"
#include "kaminpar-dist/datastructures/distributed_csr_graph.h"
#include "kaminpar-dist/datastructures/distributed_graph.h"
#include "kaminpar-dist/dkaminpar.h"
#include "kaminpar-dist/timer.h"

#include "kaminpar-shm/coarsening/max_cluster_weights.h"

#include "kaminpar-common/console_io.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/perf.h"
#include "kaminpar-common/random.h"
#include "kaminpar-common/strutils.h"
#include "kaminpar-common/timer.h"

#include "apps/io/dist_metis_parser.h"
#include "apps/io/dist_parhip_parser.h"

using namespace kaminpar;
using namespace kaminpar::dist;

namespace {

template <typename Lambda> void root_run(Lambda &&l) {
  if (mpi::get_comm_rank(MPI_COMM_WORLD) == 0) {
    l();
  }
}

template <typename Lambda> [[noreturn]] void root_run_and_exit(Lambda &&l) {
  root_run_and_exit(std::forward<Lambda>(l));
  std::exit(MPI_Finalize());
}

enum class GraphFileFormat {
  METIS,
  PARHIP
};

[[nodiscard]] std::unordered_map<std::string, GraphFileFormat> get_graph_file_formats() {
  return {
      {"metis", GraphFileFormat::METIS},
      {"parhip", GraphFileFormat::PARHIP},
  };
}

[[nodiscard]] DistributedGraph load_graph(
    const std::string &filename,
    const GraphFileFormat format,
    const GraphDistribution distribution,
    const bool graph_compression
) {

  if (graph_compression) {
    const auto read_graph = [&] {
      switch (format) {
      case GraphFileFormat::METIS:
        return io::metis::compress_read(filename, distribution, false, MPI_COMM_WORLD);
      case GraphFileFormat::PARHIP:
        return io::parhip::compressed_read(filename, distribution, false, MPI_COMM_WORLD);
      default:
        __builtin_unreachable();
      }
    };

    return DistributedGraph(std::make_unique<DistributedCompressedGraph>(read_graph()));
  } else {
    const auto read_graph = [&] {
      switch (format) {
      case GraphFileFormat::METIS:
        return io::metis::csr_read(filename, distribution, false, MPI_COMM_WORLD);
      case GraphFileFormat::PARHIP:
        return io::parhip::csr_read(filename, distribution, false, MPI_COMM_WORLD);
      default:
        __builtin_unreachable();
      }
    };

    return DistributedGraph(std::make_unique<DistributedCSRGraph>(read_graph()));
  }
}

} // namespace

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  Context ctx = create_default_context();
  ctx.partition.k = 64;

  std::string graph_filename;
  GraphFileFormat graph_file_format = GraphFileFormat::METIS;
  GraphDistribution graph_distribution = GraphDistribution::BALANCED_EDGES;

  int seed = 0;
  bool no_huge_pages = false;

  CLI::App app("Distributed-memory LP benchmark");
  app.add_option("-G,--graph", graph_filename)->required();
  app.add_option("--format", graph_file_format)
      ->transform(CLI::CheckedTransformer(get_graph_file_formats()).description(""))
      ->description(
          R"(Graph input format.
  - metis:  text format used by the Metis family
  - parhip: binary format used by ParHiP)"
      )
      ->capture_default_str();

  app.add_option("-k", ctx.partition.k)->capture_default_str();
  app.add_option("-t", ctx.parallel.num_threads)->capture_default_str();
  app.add_flag("--no-huge-pages", no_huge_pages, "Do not use huge pages via TBBmalloc.");
  app.add_option("-s", seed)->capture_default_str();

  create_coarsening_options(&app, ctx);
  create_compression_options(&app, ctx);
  CLI11_PARSE(app, argc, argv);

  Random::reseed(seed);

  tbb::global_control gc(tbb::global_control::max_allowed_parallelism, ctx.parallel.num_threads);
  scalable_allocation_mode(TBBMALLOC_USE_HUGE_PAGES, !no_huge_pages);

  auto graph =
      load_graph(graph_filename, graph_file_format, graph_distribution, ctx.compression.enabled);
  ctx.partition.graph = std::make_unique<GraphContext>(graph, ctx.partition);

  ctx.debug.graph_filename = str::extract_basename(graph_filename);
  ctx.parallel.num_mpis = mpi::get_comm_size(MPI_COMM_WORLD);
  if (ctx.compression.enabled) {
    ctx.compression.setup(graph.compressed_graph());
  }

  root_run([&] {
    cio::print_dkaminpar_banner();
    cio::print_build_identifier();
    cio::print_build_datatypes<
        NodeID,
        EdgeID,
        NodeWeight,
        EdgeWeight,
        shm::NodeWeight,
        shm::EdgeWeight>();
    cio::print_delimiter("Input Summary");
    LOG << "Execution mode:               " << ctx.parallel.num_mpis << " MPI process"
        << (ctx.parallel.num_mpis > 1 ? "es" : "") << " a " << ctx.parallel.num_threads << " thread"
        << (ctx.parallel.num_threads > 1 ? "s" : "");
  });
  print(ctx, mpi::get_comm_rank(MPI_COMM_WORLD) == 0, std::cout, graph.communicator());

  mpi::barrier(MPI_COMM_WORLD);
  GLOBAL_TIMER.reset();
  perf::start();

  GlobalLPClusterer clusterer(ctx);
  GlobalNodeWeight max_cluster_weight = shm::compute_max_cluster_weight<GlobalNodeWeight>(
      ctx.coarsening, ctx.partition, graph.global_n(), graph.global_total_node_weight()
  );
  StaticArray<GlobalNodeID> clustering(graph.total_n(), static_array::noinit);

  clusterer.set_max_cluster_weight(max_cluster_weight);
  clusterer.cluster(clustering, graph);

  // Output statistics
  mpi::barrier(MPI_COMM_WORLD);
  const std::string perf_output = perf::stop();
  STOP_TIMER();

  finalize_distributed_timer(Timer::global(), MPI_COMM_WORLD);
  root_run([&] {
    cio::print_delimiter("Result Summary");
    Timer::global().print_human_readable(std::cout);
  });

  SLOG << perf_output;

  return MPI_Finalize();
}
