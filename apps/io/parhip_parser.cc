/*******************************************************************************
 * Sequential and parallel ParHiP parser.
 *
 * @file:   parhip_parser.cc
 * @author: Daniel Salwasser
 * @date:   15.02.2024
 ******************************************************************************/
#include "apps/io/parhip_parser.h"

#include <array>
#include <cstdint>
#include <fstream>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>
#include <unistd.h>

#include "kaminpar-shm/datastructures/compressed_graph_builder.h"
#include "kaminpar-shm/graphutils/permutator.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/concurrent_circular_vector.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/parallel/loops.h"
#include "kaminpar-common/timer.h"

namespace {

class BinaryReaderException : public std::exception {
public:
  BinaryReaderException(std::string msg) : _msg(std::move(msg)) {}

  [[nodiscard]] const char *what() const noexcept override {
    return _msg.c_str();
  }

private:
  std::string _msg;
};

class BinaryReader {
public:
  BinaryReader(const std::string &filename) {
    _file = open(filename.c_str(), O_RDONLY);
    if (_file == -1) {
      throw BinaryReaderException("Cannot read the file that stores the graph");
    }

    struct stat file_info;
    if (fstat(_file, &file_info) == -1) {
      close(_file);
      throw BinaryReaderException("Cannot determine the size of the file that stores the graph");
    }

    _length = static_cast<std::size_t>(file_info.st_size);
    _data = static_cast<std::uint8_t *>(mmap(nullptr, _length, PROT_READ, MAP_PRIVATE, _file, 0));
    if (_data == MAP_FAILED) {
      close(_file);
      throw BinaryReaderException("Cannot map the file that stores the graph");
    }
  }

  ~BinaryReader() {
    munmap(_data, _length);
    close(_file);
  }

  template <typename T> [[nodiscard]] T read(std::size_t position) const {
    return *reinterpret_cast<T *>(_data + position);
  }

  template <typename T> [[nodiscard]] T *fetch(std::size_t position) const {
    return reinterpret_cast<T *>(_data + position);
  }

private:
  int _file;
  std::size_t _length;
  std::uint8_t *_data;
};

class ParhipHeader {
  using CompressedGraph = kaminpar::shm::CompressedGraph;
  using NodeID = CompressedGraph::NodeID;
  using EdgeID = CompressedGraph::EdgeID;
  using NodeWeight = CompressedGraph::NodeWeight;
  using EdgeWeight = CompressedGraph::EdgeWeight;

public:
  static constexpr std::uint64_t kSize = 3 * sizeof(std::uint64_t);

  bool has_edge_weights;
  bool has_node_weights;
  bool has_64_bit_edge_id;
  bool has_64_bit_node_id;
  bool has_64_bit_node_weight;
  bool has_64_bit_edge_weight;
  std::uint64_t num_nodes;
  std::uint64_t num_edges;

  ParhipHeader(std::uint64_t version, std::uint64_t num_nodes, std::uint64_t num_edges)
      : has_edge_weights((version & 1) == 0),
        has_node_weights((version & 2) == 0),
        has_64_bit_edge_id((version & 4) == 0),
        has_64_bit_node_id((version & 8) == 0),
        has_64_bit_node_weight((version & 16) == 0),
        has_64_bit_edge_weight((version & 32) == 0),
        num_nodes(num_nodes),
        num_edges(num_edges) {}

  void validate() const {
    if (has_64_bit_node_id) {
      if (sizeof(NodeID) != 8) {
        LOG_ERROR << "The stored graph uses 64-Bit node IDs but this build uses 32-Bit node IDs.";
        std::exit(1);
      }
    } else if (sizeof(NodeID) != 4) {
      LOG_ERROR << "The stored graph uses 32-Bit node IDs but this build uses 64-Bit node IDs.";
      std::exit(1);
    }

    if (has_64_bit_edge_id) {
      if (sizeof(EdgeID) != 8) {
        LOG_ERROR << "The stored graph uses 64-Bit edge IDs but this build uses 32-Bit edge IDs.";
        std::exit(1);
      }
    } else if (sizeof(EdgeID) != 4) {
      LOG_ERROR << "The stored graph uses 32-Bit edge IDs but this build uses 64-Bit edge IDs.";
      std::exit(1);
    }

    if (has_64_bit_node_weight) {
      if (sizeof(NodeWeight) != 8) {
        LOG_ERROR << "The stored graph uses 64-Bit node weights but this build uses 32-Bit node"
                     "weights.";
        std::exit(1);
      }
    } else if (sizeof(NodeWeight) != 4) {
      LOG_ERROR << "The stored graph uses 32-Bit node weights but this build uses 64-Bit node"
                   "weights.";
      std::exit(1);
    }

    if (has_64_bit_edge_weight) {
      if (sizeof(EdgeWeight) != 8) {
        LOG_ERROR << "The stored graph uses 64-Bit edge weights but this build uses 32-Bit edge "
                     "weights.";
        std::exit(1);
      }
    } else if (sizeof(EdgeWeight) != 4) {
      LOG_ERROR << "The stored graph uses 32-Bit edge weights but this build uses 64-Bit edge"
                   "weights.";
      std::exit(1);
    }
  }
};

} // namespace

namespace kaminpar::shm::io::parhip {

CSRGraph csr_read(const std::string &filename, const bool sorted) {
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    LOG_ERROR << "Cannot read graph stored at " << filename << ".";
    std::exit(1);
  }

  std::array<std::uint64_t, 3> raw_header;
  in.read(reinterpret_cast<char *>(raw_header.data()), ParhipHeader::kSize);

  const ParhipHeader header(raw_header[0], raw_header[1], raw_header[2]);
  header.validate();

  StaticArray<EdgeID> nodes(header.num_nodes + 1, static_array::noinit);
  in.read(reinterpret_cast<char *>(nodes.data()), (header.num_nodes + 1) * sizeof(EdgeID));

  const EdgeID nodes_offset = ParhipHeader::kSize + (header.num_nodes + 1) * sizeof(EdgeID);
  tbb::parallel_for(tbb::blocked_range<NodeID>(0, header.num_nodes + 1), [&](const auto &r) {
    for (NodeID u = r.begin(); u != r.end(); ++u) {
      nodes[u] = (nodes[u] - nodes_offset) / sizeof(NodeID);
    }
  });

  StaticArray<NodeID> edges(header.num_edges, static_array::noinit);
  in.read(reinterpret_cast<char *>(edges.data()), header.num_edges * sizeof(NodeID));

  StaticArray<NodeWeight> node_weights;
  if (header.has_node_weights) {
    node_weights.resize(header.num_nodes, static_array::noinit);
    in.read(reinterpret_cast<char *>(node_weights.data()), header.num_nodes * sizeof(NodeWeight));
  }

  StaticArray<EdgeWeight> edge_weights;
  if (header.has_edge_weights) {
    edge_weights.resize(header.num_edges, static_array::noinit);
    in.read(reinterpret_cast<char *>(edge_weights.data()), header.num_edges * sizeof(EdgeWeight));
  }

  return CSRGraph(
      std::move(nodes), std::move(edges), std::move(node_weights), std::move(edge_weights), sorted
  );
}

CompressedGraph compressed_read(const std::string &filename, const bool sorted) {
  try {
    BinaryReader reader(filename);

    const auto version = reader.read<std::uint64_t>(0);
    const auto num_nodes = reader.read<std::uint64_t>(sizeof(std::uint64_t));
    const auto num_edges = reader.read<std::uint64_t>(sizeof(std::uint64_t) * 2);
    const ParhipHeader header(version, num_nodes, num_edges);
    header.validate();

    CompressedGraphBuilder builder(
        header.num_nodes, header.num_edges, header.has_node_weights, header.has_edge_weights, sorted
    );

    std::size_t position = ParhipHeader::kSize;

    const EdgeID *nodes = reader.fetch<EdgeID>(position);
    position += (header.num_nodes + 1) * sizeof(EdgeID);

    const NodeID *edges = reader.fetch<NodeID>(position);
    position += header.num_edges + sizeof(NodeID);

    const NodeWeight *node_weights = reader.fetch<NodeWeight>(position);
    position += header.num_nodes + sizeof(NodeWeight);

    const EdgeWeight *edge_weights = reader.fetch<EdgeWeight>(position);

    // Since the offsets stored in the (raw) node array of the binary are relative byte adresses
    // into the binary itself, these offsets must be mapped to the actual edge IDs.
    const EdgeID nodes_offset_base = ParhipHeader::kSize + (header.num_nodes + 1) * sizeof(EdgeID);
    const auto map_edge_offset = [&](const NodeID node) {
      return (nodes[node] - nodes_offset_base) / sizeof(NodeID);
    };

    std::vector<std::pair<NodeID, EdgeWeight>> neighbourhood;
    for (NodeID u = 0; u < header.num_nodes; ++u) {
      const EdgeID offset = map_edge_offset(u);
      const EdgeID next_offset = map_edge_offset(u + 1);

      const auto degree = static_cast<NodeID>(next_offset - offset);
      for (NodeID i = 0; i < degree; ++i) {
        const EdgeID e = offset + i;

        const NodeID adjacent_node = edges[e];
        const EdgeWeight edge_weight = header.has_edge_weights ? edge_weights[e] : 1;

        neighbourhood.emplace_back(adjacent_node, edge_weight);
      }

      builder.add_node(u, neighbourhood);
      if (header.has_node_weights) {
        builder.add_node_weight(u, node_weights[u]);
      }

      neighbourhood.clear();
    }

    return builder.build();
  } catch (const BinaryReaderException &e) {
    LOG_ERROR << e.what();
    std::exit(1);
  }
}

CompressedGraph compressed_read_parallel(const std::string &filename, const NodeOrdering ordering) {
  try {
    BinaryReader reader(filename);

    // Read information about the graph from the header and validates whether the graph can be
    // processed.
    const auto version = reader.read<std::uint64_t>(0);
    const auto num_nodes = reader.read<std::uint64_t>(sizeof(std::uint64_t));
    const auto num_edges = reader.read<std::uint64_t>(sizeof(std::uint64_t) * 2);
    const ParhipHeader header(version, num_nodes, num_edges);
    header.validate();

    // Initializes pointers into the binary which point to the positions where the different parts
    // of the graph are stored.
    std::size_t position = ParhipHeader::kSize;

    const EdgeID *nodes = reader.fetch<EdgeID>(position);
    position += (header.num_nodes + 1) * sizeof(EdgeID);

    const NodeID *edges = reader.fetch<NodeID>(position);
    position += header.num_edges + sizeof(NodeID);

    const NodeWeight *node_weights = reader.fetch<NodeWeight>(position);
    position += header.num_nodes + sizeof(NodeWeight);

    const EdgeWeight *edge_weights = reader.fetch<EdgeWeight>(position);

    // Since the offsets stored in the (raw) node array of the binary are relative byte adresses
    // into the binary itself, these offsets must be mapped to the actual edge IDs.
    const EdgeID nodes_offset_base = ParhipHeader::kSize + (header.num_nodes + 1) * sizeof(EdgeID);
    const auto fetch_edge_offset = [&](const NodeID node) {
      return (nodes[node] - nodes_offset_base) / sizeof(NodeID);
    };
    const auto fetch_degree = [&](const NodeID node) {
      return static_cast<NodeID>((nodes[node + 1] - nodes[node]) / sizeof(NodeID));
    };

    const bool sort_by_degree_bucket = ordering == NodeOrdering::DEGREE_BUCKETS;
    if (sort_by_degree_bucket) {
      RECORD("degrees") StaticArray<NodeID> degrees(num_nodes, static_array::noinit);
      TIMED_SCOPE("Read degrees") {
        tbb::parallel_for(tbb::blocked_range<NodeID>(0, num_nodes), [&](const auto &r) {
          for (NodeID u = r.begin(); u != r.end(); ++u) {
            degrees[u] = fetch_degree(u);
          }
        });
      };
      const auto [perm, inv_perm] =
          graph::sort_by_degree_buckets(num_nodes, [&](const NodeID u) { return degrees[u]; });

      return ParallelCompressedGraphBuilder::compress(
          num_nodes,
          num_edges,
          header.has_node_weights,
          header.has_edge_weights,
          true,
          [&](const NodeID u) { return inv_perm[u]; },
          [&](const NodeID u) { return degrees[u]; },
          [&](const NodeID u) { return fetch_edge_offset(u); },
          [&](const EdgeID e) { return perm[edges[e]]; },
          [&](const NodeID u) { return node_weights[u]; },
          [&](const EdgeID e) { return edge_weights[e]; }
      );
    } else {
      return ParallelCompressedGraphBuilder::compress(
          num_nodes,
          num_edges,
          header.has_node_weights,
          header.has_edge_weights,
          ordering == NodeOrdering::IMPLICIT_DEGREE_BUCKETS,
          [](const NodeID u) { return u; },
          [&](const NodeID u) { return fetch_degree(u); },
          [&](const NodeID u) { return fetch_edge_offset(u); },
          [&](const EdgeID e) { return edges[e]; },
          [&](const NodeID u) { return node_weights[u]; },
          [&](const EdgeID e) { return edge_weights[e]; }
      );
    }
  } catch (const BinaryReaderException &e) {
    LOG_ERROR << e.what();
    std::exit(1);
  }
}

} // namespace kaminpar::shm::io::parhip
