/*******************************************************************************
 * IO utilities for the shared-memory partitioner.
 *
 * @file:   shm_io.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#include "apps/io/shm_io.h"

#include <fstream>

#include "kaminpar-shm/datastructures/compressed_graph.h"
#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/logger.h"

#include "apps/io/metis_parser.h"

namespace kaminpar::shm::io {
//
// Public Metis functions
//
namespace metis {

template <bool checked> void check_format(kaminpar::io::metis::Format format) {
  if constexpr (checked) {
    if (format.number_of_nodes >= static_cast<std::uint64_t>(std::numeric_limits<NodeID>::max())) {
      LOG_ERROR << "number of nodes is too large for the node ID type";
      std::exit(1);
    }
    if (format.number_of_edges >= static_cast<std::uint64_t>(std::numeric_limits<EdgeID>::max())) {
      LOG_ERROR << "number of edges is too large for the edge ID type";
      std::exit(1);
    }
    if (format.number_of_edges > (format.number_of_nodes * (format.number_of_nodes - 1) / 2)) {
      LOG_ERROR << "specified number of edges is impossibly large";
      std::exit(1);
    }
  } else {
    KASSERT(
        format.number_of_nodes <= static_cast<std::uint64_t>(std::numeric_limits<NodeID>::max()),
        "number of nodes is too large for the node ID type"
    );
    KASSERT(
        format.number_of_edges <= static_cast<std::uint64_t>(std::numeric_limits<EdgeID>::max()),
        "number of edges is too large for the edge ID type"
    );
    KASSERT(
        format.number_of_edges <= (format.number_of_nodes * (format.number_of_nodes - 1)) / 2,
        "specified number of edges is impossibly large"
    );
  }
}

template <bool checked> void check_node_weight(const std::uint64_t weight) {
  if constexpr (checked) {
    if (weight > static_cast<std::uint64_t>(std::numeric_limits<NodeWeight>::max())) {
      LOG_ERROR << "node weight is too large for the node weight type";
      std::exit(1);
    }
    if (weight <= 0) {
      LOG_ERROR << "zero node weights are not supported";
      std::exit(1);
    }
  } else {
    KASSERT(
        weight <= static_cast<std::uint64_t>(std::numeric_limits<NodeWeight>::max()),
        "node weight is too large for the node weight type"
    );
    KASSERT(weight > 0u, "zero node weights are not supported");
  }
}

template <bool checked>
void check_edge(
    const std::uint64_t node_count,
    const std::uint64_t u,
    const std::uint64_t weight,
    const std::uint64_t v
) {
  if constexpr (checked) {
    if (weight > static_cast<std::uint64_t>(std::numeric_limits<EdgeWeight>::max())) {
      LOG_ERROR << "edge weight is too large for the edge weight type";
      std::exit(1);
    }
    if (weight <= 0) {
      LOG_ERROR << "zero edge weights are not supported";
      std::exit(1);
    }
    if (v + 1 >= node_count) {
      LOG_ERROR << "neighbor " << v + 1 << " of nodes " << u + 1 << " is out of bounds";
      std::exit(1);
    }
    if (v + 1 == u) {
      LOG_ERROR << "detected self-loop on node " << v + 1 << ", which is not allowed";
      std::exit(1);
    }
  } else {
    KASSERT(
        weight <= static_cast<std::uint64_t>(std::numeric_limits<EdgeWeight>::max()),
        "edge weight is too large for the edge weight type"
    );
    KASSERT(weight > 0u, "zero edge weights are not supported");
    KASSERT(v + 1 < node_count, "neighbor out of bounds");
    KASSERT(u != v + 1, "detected illegal self-loop");
  }
}

template <bool checked>
void check_total_weight(std::int64_t total_node_weight, std::int64_t total_edge_weight) {
  if constexpr (checked) {
    if (total_node_weight > static_cast<std::int64_t>(std::numeric_limits<NodeWeight>::max())) {
      LOG_ERROR << "total node weight does not fit into the node weight type";
      std::exit(1);
    }
    if (total_edge_weight > static_cast<std::int64_t>(std::numeric_limits<EdgeWeight>::max())) {
      LOG_ERROR << "total edge weight does not fit into the edge weight type";
      std::exit(1);
    }
  } else {
    KASSERT(
        total_node_weight <= static_cast<std::int64_t>(std::numeric_limits<NodeWeight>::max()),
        "total node weight does not fit into the node weight type"
    );
    KASSERT(
        total_edge_weight <= static_cast<std::int64_t>(std::numeric_limits<EdgeWeight>::max()),
        "total edge weight does not fit into the edge weight type"
    );
  }
}

template <bool checked>
void read(
    const std::string &filename,
    StaticArray<EdgeID> &nodes,
    StaticArray<NodeID> &edges,
    StaticArray<NodeWeight> &node_weights,
    StaticArray<EdgeWeight> &edge_weights
) {
  using namespace kaminpar::io::metis;

  bool store_node_weights = false;
  bool store_edge_weights = false;
  std::int64_t total_node_weight = 0;
  std::int64_t total_edge_weight = 0;

  NodeID u = 0;
  EdgeID e = 0;

  parse<false>(
      filename,
      [&](const auto &format) {
        check_format<checked>(format);

        store_node_weights = format.has_node_weights;
        store_edge_weights = format.has_edge_weights;
        nodes.resize(format.number_of_nodes + 1);
        edges.resize(format.number_of_edges * 2);
        if (store_node_weights) {
          node_weights.resize(format.number_of_nodes);
        }
        if (store_edge_weights) {
          edge_weights.resize(format.number_of_edges * 2);
        }
      },
      [&](const std::uint64_t weight) {
        check_node_weight<checked>(weight);

        if (store_node_weights) {
          node_weights[u] = static_cast<NodeWeight>(weight);
        }
        nodes[u] = e;
        total_node_weight += weight;
        ++u;
      },
      [&](const std::uint64_t weight, const std::uint64_t v) {
        check_edge<checked>(nodes.size(), u, weight, v);

        if (store_edge_weights) {
          edge_weights[e] = static_cast<EdgeWeight>(weight);
        }
        edges[e] = static_cast<NodeID>(v);
        total_edge_weight += weight;
        ++e;
      }
  );
  nodes[u] = e;

  check_total_weight<checked>(total_edge_weight, total_edge_weight);

  // only keep weights if the graph is really weighted
  const bool unit_node_weights = static_cast<NodeID>(total_node_weight + 1) == nodes.size();
  if (unit_node_weights) {
    node_weights.free();
  }

  const bool unit_edge_weights = static_cast<EdgeID>(total_edge_weight) == edges.size();
  if (unit_edge_weights) {
    edge_weights.free();
  }
}

template void read<false>(
    const std::string &filename,
    StaticArray<EdgeID> &nodes,
    StaticArray<NodeID> &edges,
    StaticArray<NodeWeight> &node_weights,
    StaticArray<EdgeWeight> &edge_weights
);

template void read<true>(
    const std::string &filename,
    StaticArray<EdgeID> &nodes,
    StaticArray<NodeID> &edges,
    StaticArray<NodeWeight> &node_weights,
    StaticArray<EdgeWeight> &edge_weights
);

template <bool checked> CSRGraph csr_read(const std::string &filename) {
  using namespace kaminpar::io::metis;

  StaticArray<EdgeID> nodes;
  StaticArray<NodeID> edges;
  StaticArray<NodeWeight> node_weights;
  StaticArray<EdgeWeight> edge_weights;

  bool store_node_weights = false;
  bool store_edge_weights = false;

  std::int64_t total_node_weight = 0;
  std::int64_t total_edge_weight = 0;

  NodeID u = 0;
  EdgeID e = 0;

  parse<false>(
      filename,
      [&](const auto &format) {
        check_format<checked>(format);

        store_node_weights = format.has_node_weights;
        store_edge_weights = format.has_edge_weights;

        nodes.resize(format.number_of_nodes + 1);
        edges.resize(format.number_of_edges * 2);

        if (store_node_weights) {
          node_weights.resize(format.number_of_nodes);
        }

        if (store_edge_weights) {
          edge_weights.resize(format.number_of_edges * 2);
        }
      },
      [&](const std::uint64_t weight) {
        check_node_weight<checked>(weight);

        total_node_weight += weight;
        if (store_node_weights) {
          node_weights[u] = static_cast<NodeWeight>(weight);
        }

        nodes[u] = e;
        u += 1;
      },
      [&](const std::uint64_t weight, const std::uint64_t v) {
        check_edge<checked>(nodes.size(), u, weight, v);

        total_edge_weight += weight;
        if (store_edge_weights) {
          edge_weights[e] = static_cast<EdgeWeight>(weight);
        }

        edges[e] = static_cast<NodeID>(v);
        e += 1;
      }
  );
  nodes[u] = e;

  check_total_weight<checked>(total_node_weight, total_edge_weight);

  // Only keep weights if the graph is really weighted.
  const bool unit_node_weights = static_cast<NodeID>(total_node_weight + 1) == nodes.size();
  if (unit_node_weights) {
    node_weights.free();
  }

  const bool unit_edge_weights = static_cast<EdgeID>(total_edge_weight) == edges.size();
  if (unit_edge_weights) {
    edge_weights.free();
  }

  return CSRGraph(
      std::move(nodes), std::move(edges), std::move(node_weights), std::move(edge_weights)
  );
}

template CSRGraph csr_read<false>(const std::string &filename);
template CSRGraph csr_read<true>(const std::string &filename);

template <bool checked> CompressedGraph compress_read(const std::string &filename) {
  using namespace kaminpar::io::metis;

  std::uint64_t number_of_nodes = 0;
  bool store_node_weights = false;
  bool store_edge_weights = false;

  NodeID node = 0;
  EdgeID edge = 0;

  CompressedGraphBuilder builder;
  std::vector<NodeID> neighbourhood;

  parse<false>(
      filename,
      [&](const auto &format) {
        check_format<checked>(format);

        number_of_nodes = format.number_of_nodes + 1;
        store_node_weights = format.has_node_weights;
        store_edge_weights = format.has_edge_weights;

        builder.init(
            format.number_of_nodes, format.number_of_edges, store_node_weights, store_edge_weights
        );
      },
      [&](const std::uint64_t weight) {
        check_node_weight<checked>(weight);

        if (node > 0) {
          builder.add_node(node - 1, neighbourhood);
          neighbourhood.clear();
        }

        if (store_node_weights) {
          builder.set_node_weight(node, static_cast<NodeWeight>(weight));
        }

        node += 1;
      },
      [&](const std::uint64_t weight, const std::uint64_t v) {
        check_edge<checked>(number_of_nodes, node, weight, v);

        if (store_edge_weights) {
          builder.set_edge_weight(node, static_cast<EdgeWeight>(weight));
        }

        neighbourhood.push_back(static_cast<NodeID>(v));
        edge += 1;
      }
  );
  builder.add_node(node - 1, neighbourhood);

  check_total_weight<checked>(builder.total_node_weight(), builder.total_edge_weight());

  return builder.build();
}

template CompressedGraph compress_read<false>(const std::string &filename);
template CompressedGraph compress_read<true>(const std::string &filename);

} // namespace metis

//
// Partition
//

namespace partition {
void write(const std::string &filename, const std::vector<BlockID> &partition) {
  std::ofstream out(filename);
  for (const BlockID block : partition) {
    out << block << "\n";
  }
}

std::vector<BlockID> read(const std::string &filename) {
  using namespace kaminpar::io;

  MappedFileToker<> toker(filename);
  std::vector<BlockID> partition;
  while (toker.valid_position()) {
    partition.push_back(toker.scan_uint());
    toker.consume_char('\n');
  }

  return partition;
}
} // namespace partition
} // namespace kaminpar::shm::io
