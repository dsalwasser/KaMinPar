/*******************************************************************************
 * Graph properties tool for the shared-memory algorithm.
 *
 * @file:   shm_graph_properties_tool.cc
 * @author: Daniel Salwasser
 * @date:   26.12.2023
 ******************************************************************************/
// clang-format off
#include <kaminpar-cli/kaminpar_arguments.h>
// clang-format on

#include <tbb/global_control.h>

#include "kaminpar-io/kaminpar_io.h"

#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/metrics.h"

#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/strutils.h"

using namespace kaminpar;
using namespace kaminpar::shm;

int main(int argc, char *argv[]) {
  Context ctx = create_default_context();
  std::string graph_filename;
  std::string partition_filename;
  std::string block_sizes_filename;

  io::GraphFileFormat graph_file_format = io::GraphFileFormat::METIS;

  CLI::App app("Shared-memory partition properties tool");
  app.add_option("-G,--graph", graph_filename, "Input graph")->required();

  auto *partition_group = app.group("Partition options:"); //->require_option(1);
  partition_group->add_option(
      "-P,--partition", partition_filename, "Partition (block of one node per line)"
  );
  partition_group->add_option(
      "--block-sizes", block_sizes_filename, "Block sizes (one size per line)"
  );

  app.add_option("-t,--threads", ctx.parallel.num_threads, "Number of threads");
  app.add_option("-f,--graph-file-format", graph_file_format)
      ->transform(CLI::CheckedTransformer(
          std::unordered_map<std::string, io::GraphFileFormat>{
              {"metis", io::GraphFileFormat::METIS},
              {"parhip", io::GraphFileFormat::PARHIP},
              {"compressed", io::GraphFileFormat::COMPRESSED},
          },
          CLI::ignore_case
      ))
      ->description(R"(Graph file formats:
  - metis
  - parhip
  - compressed)");
  create_graph_compression_options(&app, ctx);
  CLI11_PARSE(app, argc, argv);

  tbb::global_control gc(tbb::global_control::max_allowed_parallelism, ctx.parallel.num_threads);

  auto graph = io::read_graph(graph_filename, graph_file_format, ctx.compression.enabled);
  if (!graph) {
    LOG_ERROR << "Failed to read the input graph";
    return EXIT_FAILURE;
  }

  ctx.debug.graph_name = str::extract_basename(graph_filename);
  ctx.compression.setup(*graph);

  LOG << "Graph:            " << ctx.debug.graph_name;

  std::vector<BlockID> partition;
  if (!partition_filename.empty()) {
    LOG << "Partition:        " << str::extract_basename(partition_filename);

    partition = io::read_partition(partition_filename);
  } else if (!block_sizes_filename.empty()) {
    LOG << "Block sizes:      " << str::extract_basename(block_sizes_filename);

    partition = io::read_block_sizes(block_sizes_filename);
  } else {
    LOG_ERROR << "No partition or block sizes provided";
    return EXIT_FAILURE;
  }

  const BlockID k = *std::max_element(partition.begin(), partition.end()) + 1;
  PartitionedGraph p_graph(*graph, k, StaticArray<BlockID>(partition.size(), partition.data()));

  LOG << "Number of blocks: " << k;
  LOG << "Edge cut:         " << metrics::edge_cut(p_graph);
  LOG << "Imbalance:        " << metrics::imbalance(p_graph);

  return EXIT_SUCCESS;
}
