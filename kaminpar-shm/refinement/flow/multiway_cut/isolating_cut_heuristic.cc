#include "kaminpar-shm/refinement/flow/multiway_cut/isolating_cut_heuristic.h"

#include <algorithm>
#include <limits>
#include <queue>
#include <utility>

#include "kaminpar.h"

#include "kaminpar-shm/refinement/flow/max_flow/edmond_karp_algorithm.h"
#include "kaminpar-shm/refinement/flow/max_flow/fifo_preflow_push_algorithm.h"

#include "kaminpar-common/assert.h"
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
    const PartitionedCSRGraph &p_graph,
    const CSRGraph &graph,
    std::span<const NodeID> reverse_edges,
    const TerminalSets &terminal_sets
) {
  _p_graph = &p_graph;
  _graph = &graph;
  _reverse_edges = reverse_edges;

  if (_isolating_assignment.size() < graph.n()) {
    _isolating_assignment.resize(graph.n(), static_array::noinit);
  }
  std::fill_n(_isolating_assignment.begin(), graph.n(), kInvalidBlockID);

  std::unordered_set<EdgeID> cut_edges;
  BlockID heaviest_side = kInvalidBlockID;
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
      heaviest_side = terminal_set;
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

  repair_isolating_assignment(
      heaviest_side, terminal_sets.terminal_set_nodes(heaviest_side), cut_edges
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
    KASSERT(_isolating_assignment[u] == kInvalidBlockID);
    _isolating_assignment[u] = block;

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

void IsolatingCutHeuristic::repair_isolating_assignment(
    const BlockID block,
    std::span<const NodeID> terminals,
    const std::unordered_set<EdgeID> &cut_edges
) {
  SCOPED_TIMER("Repair Isolating Assignment");

  std::unordered_set<NodeID> terminal_side_nodes;
  std::queue<NodeID> bfs_queue;
  for (const NodeID terminal : terminals) {
    terminal_side_nodes.insert(terminal);
    bfs_queue.push(terminal);
  }

  while (!bfs_queue.empty()) {
    const NodeID u = bfs_queue.front();
    bfs_queue.pop();

    _graph->neighbors(u, [&](const EdgeID e, const NodeID v) {
      if (terminal_side_nodes.contains(v) || cut_edges.contains(e)) {
        return;
      }

      _isolating_assignment[v] = block;
      terminal_side_nodes.insert(v);
      bfs_queue.push(v);
    });
  }
}

EdgeWeight
IsolatingCutHeuristic::compute_cut_value(const std::unordered_set<EdgeID> &cut_edges) const {
  SCOPED_TIMER("Compute Cut Value");

  EdgeWeight cut_value = 0;
  for (const NodeID u : _graph->nodes()) {
    const BlockID u_block = _p_graph->block(u);
    const BlockID u_isolating_block = _isolating_assignment[u];
    const bool u_reachable = u_isolating_block != kInvalidBlockID;

    _graph->neighbors(u, [&](const EdgeID e, const NodeID v, const EdgeWeight w) {
      const BlockID v_block = _p_graph->block(v);
      const BlockID v_isolating_block = _isolating_assignment[v];

      const bool cut_edge = cut_edges.contains(e);
      const bool v_reachable = v_isolating_block != kInvalidBlockID;

      if (cut_edge) {
        if (u_reachable && v_reachable) {
          cut_value += w;
        } else if (u_reachable) {
          cut_value += (u_isolating_block != v_block) ? w : 0;
        } else if (v_reachable) {
          cut_value += (v_isolating_block != u_block) ? w : 0;
        } else {
          cut_value += (u_block != v_block) ? w : 0;
        }
      }
    });
  }

  return cut_value / 2;
}

} // namespace kaminpar::shm
