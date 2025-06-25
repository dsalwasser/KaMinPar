#include "kaminpar-shm/refinement/flow/multiway_cut/isolating_cut_heuristic.h"

#include <algorithm>
#include <limits>
#include <queue>
#include <unordered_set>
#include <utility>

#include "kaminpar.h"

#include "kaminpar-shm/refinement/flow/max_flow/edmond_karp_algorithm.h"
#include "kaminpar-shm/refinement/flow/max_flow/fifo_preflow_push_algorithm.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

IsolatingCutHeuristic::IsolatingCutHeuristic(const IsolatingCutHeuristicContext &ctx) : _ctx(ctx) {
  switch (ctx.flow_algorithm) {
  case FlowAlgorithm::EDMONDS_KARP:
    _max_flow_algorithm = std::make_unique<EdmondsKarpAlgorithm>();
    break;
  case FlowAlgorithm::FIFO_PREFLOW_PUSH:
    _max_flow_algorithm = std::make_unique<FIFOPreflowPushAlgorithm>(ctx.fifo_preflow_push);
    break;
  }
}

MultiwayCutAlgorithm::Result IsolatingCutHeuristic::compute(
    [[maybe_unused]] const PartitionedCSRGraph &p_graph,
    const CSRGraph &graph,
    std::span<const NodeID> reverse_edges,
    const TerminalSets &terminal_sets
) {
  _graph = &graph;
  _reverse_edges = reverse_edges;

  if (_node_assignment.size() < graph.n()) {
    _node_assignment.resize(graph.n(), static_array::noinit);
  }
  std::fill_n(_node_assignment.begin(), graph.n(), kInvalidBlockID);

  std::unordered_set<EdgeID> cut_edges;
  Cut cur_max_weighted_cut(std::numeric_limits<EdgeWeight>::min(), {});

  std::unordered_set<NodeID> terminals;
  std::unordered_set<NodeID> other_terminals;
  for (BlockID terminal_set = 0; terminal_set < terminal_sets.num_terminal_sets(); ++terminal_set) {
    for (const NodeID terminal : terminal_sets.terminal_set_nodes(terminal_set)) {
      other_terminals.insert(terminal);
    }
  }

  for (BlockID terminal_set = 0; terminal_set < terminal_sets.num_terminal_sets(); ++terminal_set) {
    terminals.clear();
    for (const NodeID terminal : terminal_sets.terminal_set_nodes(terminal_set)) {
      terminals.insert(terminal);
      other_terminals.erase(terminal);
    }

    const auto [flow_value, flow] = TIMED_SCOPE("Compute Max Flow") {
      _max_flow_algorithm->initialize(graph, _reverse_edges, terminals, other_terminals);
      return _max_flow_algorithm->compute_max_flow();
    };

    Cut cut = compute_cut(terminal_set, terminals, flow);
    KASSERT(flow_value == cut.value);

    if (cut.value > cur_max_weighted_cut.value) {
      std::swap(cut, cur_max_weighted_cut);
    }

    for (const EdgeID cut_edge : cut.edges) {
      cut_edges.insert(cut_edge);
    }

    for (const NodeID terminal : terminal_sets.terminal_set_nodes(terminal_set)) {
      other_terminals.insert(terminal);
    }
  }

  KASSERT(
      debug::is_valid_multiway_cut(graph, terminal_sets, cut_edges),
      "computed a non-valid multi-way cut using the isolating-cut heuristic",
      assert::heavy
  );

  const EdgeWeight cut_value = compute_cut_value(cut_edges);
  return Result(cut_value, std::move(cut_edges));
}

IsolatingCutHeuristic::Cut IsolatingCutHeuristic::compute_cut(
    const BlockID block,
    const std::unordered_set<NodeID> &terminals,
    std::span<const EdgeWeight> flow
) {
  SCOPED_TIMER("Compute Cut");

  std::unordered_set<NodeID> terminal_side_nodes;
  std::queue<NodeID> bfs_queue;
  for (const NodeID terminal : terminals) {
    terminal_side_nodes.insert(terminal);
    bfs_queue.push(terminal);
  }

  while (!bfs_queue.empty()) {
    const NodeID u = bfs_queue.front();
    bfs_queue.pop();

    _graph->neighbors(u, [&](const EdgeID e, const NodeID v, const EdgeWeight w) {
      if (terminal_side_nodes.contains(v) || flow[e] == w) {
        return;
      }

      terminal_side_nodes.insert(v);
      bfs_queue.push(v);
    });
  }

  EdgeWeight cut_value = 0;
  std::unordered_set<EdgeID> cut_edges;
  for (const NodeID u : terminal_side_nodes) {
    _node_assignment[u] = block;

    _graph->neighbors(u, [&](const EdgeID e, const NodeID v, const EdgeWeight w) {
      if (terminal_side_nodes.contains(v)) {
        return;
      }

      cut_value += w;
      cut_edges.insert(e);
      cut_edges.insert(_reverse_edges[e]);
    });
  }

  return Cut(cut_value, std::move(cut_edges));
}

EdgeWeight IsolatingCutHeuristic::compute_cut_value(const std::unordered_set<EdgeID> &cut_edges) {
  SCOPED_TIMER("Compute Cut Value");

  EdgeWeight cut_value = 0;
  for (const NodeID u : _graph->nodes()) {
    _graph->neighbors(u, [&](const EdgeID e, [[maybe_unused]] const NodeID v, const EdgeWeight w) {
      cut_value += cut_edges.contains(e) ? w : 0;
    });
  }

  return cut_value / 2;
}

} // namespace kaminpar::shm
