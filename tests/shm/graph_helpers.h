#pragma once

#include <vector>

#include <gmock/gmock.h>

#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm::testing {
//
// Convenience functions to create Graph / PartitionedGraph from initializer
// lists
//

inline Graph make_graph(
    const std::vector<EdgeID> &nodes, const std::vector<NodeID> &edges, const bool sorted = false
) {
  return Graph(std::make_unique<CSRGraph>(
      static_array::create_from(nodes),
      static_array::create_from(edges),
      StaticArray<NodeWeight>(),
      StaticArray<EdgeWeight>(),
      sorted
  ));
}

inline Graph make_graph(
    const std::vector<EdgeID> &nodes,
    const std::vector<NodeID> &edges,
    const std::vector<NodeWeight> &node_weights,
    const std::vector<EdgeWeight> &edge_weights,
    const bool sorted = false
) {
  return Graph(std::make_unique<CSRGraph>(
      static_array::create_from(nodes),
      static_array::create_from(edges),
      static_array::create_from(node_weights),
      static_array::create_from(edge_weights),
      sorted
  ));
}

inline PartitionedGraph
make_p_graph(const Graph &graph, const BlockID k, const std::vector<BlockID> &partition) {
  return PartitionedGraph(graph, k, StaticArray<BlockID>::create(partition));
}

inline PartitionedGraph make_p_graph(const Graph &graph, const BlockID k) {
  return PartitionedGraph(graph, k);
}

inline EdgeID find_edge_by_endpoints(const Graph &graph, const NodeID u, const NodeID v) {
  EdgeID ans = kInvalidEdgeID;
  graph.neighbors(u, [&](const EdgeID e, const NodeID v_prime) {
    if (ans == kInvalidEdgeID && v == v_prime) {
      ans = e;
    }
  });
  return ans;
}

inline std::vector<NodeID> degrees(const Graph &graph) {
  std::vector<NodeID> degrees(graph.n());
  for (const NodeID u : graph.nodes()) {
    degrees[u] = graph.degree(u);
  }
  return degrees;
}

inline void change_node_weight(Graph &graph, const NodeID u, const NodeWeight new_node_weight) {
  auto &raw_graph = *dynamic_cast<CSRGraph *>(graph.underlying_graph());
  auto &node_weights = raw_graph.raw_node_weights();
  node_weights[u] = new_node_weight;
}
} // namespace kaminpar::shm::testing
