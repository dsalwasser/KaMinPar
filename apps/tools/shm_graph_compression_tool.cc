/*******************************************************************************
 * Graph compression tool for the shared-memory algorithm.
 *
 * @file:   shm_graph_compression_tool.cc
 * @author: Daniel Salwasser
 * @date:   14.12.2023
 ******************************************************************************/
// clang-format off
#include <kaminpar-cli/kaminpar_arguments.h>
// clang-format on

#include <tbb/global_control.h>

#include "kaminpar-shm/context_io.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/logger.h"

#include "apps/io/shm_compressed_graph_binary.h"
#include "apps/io/shm_io.h"

using namespace kaminpar;
using namespace kaminpar::shm;
using namespace kaminpar::shm::io;

int main(int argc, char *argv[]) {
  CLI::App app("Shared-memory graph compression tool");

  std::string graph_filename;
  app.add_option("-G,--graph", graph_filename, "Input graph in METIS format")->required();

  GraphFileFormat graph_file_format = io::GraphFileFormat::METIS;
  app.add_option("-f,--graph-file-format", graph_file_format)
      ->transform(CLI::CheckedTransformer(io::get_graph_file_formats()).description(""))
      ->description(R"(Graph file formats:
  - metis
  - parhip)");

  std::string compressed_graph_filename;
  app.add_option("--out", compressed_graph_filename, "Ouput file for saving the compressed graph")
      ->required();

  NodeOrdering node_ordering = NodeOrdering::NATURAL;
  app.add_option("--node-order", node_ordering)
      ->transform(CLI::CheckedTransformer(get_node_orderings()).description(""))
      ->description(R"(Criteria by which the nodes of the graph are sorted and rearranged:
  - natural:     keep node order of the graph (do not rearrange)
  - deg-buckets: sort nodes by degree bucket and rearrange accordingly
  - implicit-deg-buckets: nodes of the input graph are sorted by deg-buckets order)")
      ->capture_default_str();

  int num_threads = 1;
  app.add_option("-t,--threads", num_threads, "Number of threads");
  CLI11_PARSE(app, argc, argv);

  tbb::global_control gc(tbb::global_control::max_allowed_parallelism, num_threads);

  LOG << "Reading input graph...";
  CompressedGraph graph = compressed_read(graph_filename, graph_file_format, node_ordering);

  LOG << "Writing compressed graph...";
  io::compressed_binary::write(compressed_graph_filename, graph);

  return 0;
}
