#include "kaminpar-shm/refinement/flow/max_flow/max_flow_algorithm.h"

#include <cstddef>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "kaminpar-common/assert.h"
#include "kaminpar-common/logger.h"

namespace kaminpar::shm {

StaticArray<EdgeWeight> MaxFlowAlgorithm::compute(
    const CSRGraph &graph,
    const std::unordered_set<NodeID> &sources,
    const std::unordered_set<NodeID> &sinks
) {
  StaticArray<EdgeWeight> flow(graph.m());
  compute(graph, sources, sinks, flow);
  return flow;
}

namespace debug {

bool are_terminals_disjoint(
    const std::unordered_set<NodeID> &sources, const std::unordered_set<NodeID> &sinks
) {
  return std::all_of(sources.begin(), sources.end(), [&](const NodeID u) {
    return !sinks.contains(u);
  });
}

bool is_valid_flow(
    const CSRGraph &graph,
    const std::unordered_set<NodeID> &sources,
    const std::unordered_set<NodeID> &sinks,
    std::span<const EdgeWeight> flow
) {
  if (graph.m() != flow.size()) {
    LOG_WARNING << "Flow size does not equal the number of edges";
    return false;
  }

  struct EdgeHasher {
    [[nodiscard]] std::size_t operator()(const std::pair<NodeID, NodeID> &edge) const noexcept {
      return edge.first ^ (edge.second << 1);
    }
  };
  std::unordered_map<std::pair<NodeID, NodeID>, EdgeID, EdgeHasher> edge_flow_table;
  std::vector<NodeID> flow_excess(graph.n(), 0);

  bool is_valid = true;

  for (const NodeID u : graph.nodes()) {
    graph.neighbors(u, [&](const EdgeID e, const NodeID v, const EdgeWeight w) {
      const EdgeWeight e_flow = flow[e];
      if (e_flow > w) {
        LOG_WARNING << "Capacity constraint violated for edge " << u << " -> " << v;
        is_valid = false;
      }

      if (u < v) {
        edge_flow_table[{u, v}] = e;
      } else {
        auto it = edge_flow_table.find({v, u});
        KASSERT(it != edge_flow_table.end());

        const EdgeID e_reverse = it->second;
        const EdgeWeight e_reverse_flow = flow[e_reverse];
        if (e_flow != -e_reverse_flow) {
          LOG_WARNING << "Antisymmetry constraint violated for edge " << v << " <-> " << u;
          is_valid = false;
        }

        edge_flow_table.erase(it);
      }

      if (e_flow > 0) {
        flow_excess[u] -= flow[e];
        flow_excess[v] += flow[e];
      }
    });
  }

  for (const NodeID u : graph.nodes()) {
    if (sources.contains(u) || sinks.contains(u)) {
      continue;
    }

    if (flow_excess[u] != 0) {
      LOG_WARNING << "Conservation constraint violated for node " << u << ' ' << flow_excess[u];
      is_valid = false;
    }
  }

  return is_valid;
}

bool is_max_flow(
    const CSRGraph &graph,
    const std::unordered_set<NodeID> &sources,
    const std::unordered_set<NodeID> &sinks,
    std::span<const EdgeWeight> flow
) {
  if (graph.m() != flow.size()) {
    LOG_WARNING << "Flow size does not equal the number of edges";
    return false;
  }

  std::unordered_set<NodeID> visited;
  std::queue<NodeID> bfs_queue;
  for (const NodeID source : sources) {
    bfs_queue.push(source);
    visited.insert(source);
  }

  while (!bfs_queue.empty()) {
    const NodeID u = bfs_queue.front();
    bfs_queue.pop();

    if (sinks.contains(u)) {
      return false;
    }

    graph.neighbors(u, [&](const EdgeID e, const NodeID v, const EdgeWeight w) {
      if (visited.contains(v)) {
        return;
      }

      const EdgeWeight e_flow = flow[e];
      if (e_flow < w) {
        bfs_queue.push(v);
        visited.insert(v);
      }
    });
  }

  return true;
}

void print_flow(
    const CSRGraph &graph,
    const std::unordered_set<NodeID> &sources,
    const std::unordered_set<NodeID> &sinks,
    std::span<const EdgeWeight> flow
) {
  for (const NodeID u : graph.nodes()) {
    graph.neighbors(u, [&](const EdgeID e, const NodeID v, const EdgeWeight c) {
      std::cout << u << " -> " << v << ": " << flow[e] << "/" << c;

      if (sources.contains(u)) {
        std::cout << " (source)";
      } else if (sinks.contains(u)) {
        std::cout << " (sink)";
      }

      std::cout << '\n';
    });
  }
}

} // namespace debug

} // namespace kaminpar::shm
