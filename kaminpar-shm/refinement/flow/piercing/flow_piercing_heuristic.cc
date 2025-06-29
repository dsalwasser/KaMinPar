#include "kaminpar-shm/refinement/flow/piercing/flow_piercing_heuristic.h"

#include "kaminpar-shm/refinement/flow/util/reverse_edge_index.h"

#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

FlowPiercingHeuristic::FlowPiercingHeuristic(const MultiwayPiercingHeuristicContext &ctx)
    : _ctx(ctx),
      _max_flow_algorithm_ctx(true, 1.0),
      _max_flow_algorithm(_max_flow_algorithm_ctx) {}

void FlowPiercingHeuristic::initialize(
    const CSRGraph &graph,
    [[maybe_unused]] const PartitionedCSRGraph &p_graph,
    const PartitionContext &p_ctx,
    const TerminalSets &terminal_sets
) {
  _graph = &graph;
  _terminal_sets = &terminal_sets;

  _reverse_edges = compute_reverse_edge_index(graph);

  _piercing_nodes_candidates_marker.resize(graph.n());

  _node_status.initialize(graph.n());

  _bulk_piercing_ctx.resize(
      terminal_sets.num_terminal_sets(),
      BulkPiercingContext(_ctx.bulk_piercing_round_threshold, _ctx.bulk_piercing_shrinking_factor)
  );

  const NodeWeight total_node_weight = graph.total_node_weight();
  const BlockWeight max_total_weight = p_ctx.total_max_block_weights(0, p_ctx.k);
  for (BlockID terminal_set = 0; terminal_set < terminal_sets.num_terminal_sets(); ++terminal_set) {
    NodeWeight side_weight = 0;
    for (const NodeID u : terminal_sets.terminal_set_nodes(terminal_set)) {
      side_weight += _graph->node_weight(u);
    }

    _bulk_piercing_ctx[terminal_set].initialize(
        side_weight, total_node_weight, p_ctx.max_block_weight(terminal_set), max_total_weight
    );
  }
}

std::span<const NodeID> FlowPiercingHeuristic::find_piercing_nodes(
    const BlockID terminal_set, const NodeWeight terminal_set_weight, const NodeWeight max_weight
) {
  _terminal_set_nodes.clear();
  _other_terminal_set_nodes.clear();

  for (const NodeID u : _terminal_sets->terminal_set_nodes(terminal_set)) {
    _terminal_set_nodes.push_back(u);
  }

  for (BlockID other_terminal_set = 0; other_terminal_set < _terminal_sets->num_terminal_sets();
       ++other_terminal_set) {
    if (other_terminal_set == terminal_set) {
      continue;
    }

    for (const NodeID u : _terminal_sets->terminal_set_nodes(other_terminal_set)) {
      _other_terminal_set_nodes.push_back(u);
    }
  }

  _max_flow_algorithm.initialize(
      *_graph, _reverse_edges, _terminal_set_nodes, _other_terminal_set_nodes
  );
  const auto [_, flow] = _max_flow_algorithm.compute_max_flow();

  _node_status.reset();
  expand_cut(true, flow, _terminal_set_nodes);
  expand_cut(false, flow, _other_terminal_set_nodes);

  compute_piercing_node_candidates();

  compute_piercing_nodes(terminal_set, terminal_set_weight, max_weight);
  return _piercing_nodes;
}

void FlowPiercingHeuristic::expand_cut(
    const bool source_side, std::span<const EdgeWeight> flow, std::span<const NodeID> terminals
) {
  SCOPED_TIMER("Expand Cut");

  std::queue<NodeID> bfs_queue;
  for (const NodeID terminal : terminals) {
    bfs_queue.push(terminal);

    if (source_side) {
      _node_status.add_source(terminal);
    } else {
      _node_status.add_sink(terminal);
    }
  }

  while (!bfs_queue.empty()) {
    const NodeID u = bfs_queue.front();
    bfs_queue.pop();

    _graph->neighbors(u, [&](const EdgeID e, const NodeID v, const EdgeWeight c) {
      if (!_node_status.is_unknown(v)) {
        return;
      }

      const EdgeWeight e_flow = flow[e];
      const bool has_residual_capacity = source_side ? (e_flow < c) : (-e_flow < c);
      if (has_residual_capacity) {
        bfs_queue.push(v);

        if (source_side) {
          _node_status.add_source(v);
        } else {
          _node_status.add_sink(v);
        }
      }
    });
  }
}

void FlowPiercingHeuristic::compute_piercing_node_candidates() {
  SCOPED_TIMER("Compute Piercing Node Candidates");

  _piercing_nodes_candidates_marker.reset();
  _unreachable_piercing_node_candidates.clear();
  _reachable_piercing_node_candidates.clear();

  for (const NodeID u : _terminal_set_nodes) {
    _graph->adjacent_nodes(u, [&](const NodeID v) {
      if (_terminal_sets->is_terminal(v)) {
        return;
      }

      if (!_piercing_nodes_candidates_marker.get(v)) {
        _piercing_nodes_candidates_marker.set(v);

        const bool reachable = _node_status.has_status(v, NodeStatus::kSink);
        if (reachable) {
          _reachable_piercing_node_candidates.push_back(v);
        } else {
          _unreachable_piercing_node_candidates.push_back(v);
        }
      }
    });
  }
}

void FlowPiercingHeuristic::compute_piercing_nodes(
    const BlockID terminal_set, const NodeWeight terminal_set_weight, const NodeWeight max_weight
) {
  SCOPED_TIMER("Compute Piercing Nodes");

  _piercing_nodes.clear();

  NodeWeight cur_weight = 0;
  const auto add_piercing_nodes = [&](const auto &piercing_node_candidates,
                                      const auto max_num_piercing_nodes) {
    for (const NodeID u : piercing_node_candidates) {
      const NodeWeight u_weight = _graph->node_weight(u);
      if (cur_weight + u_weight > max_weight) {
        continue;
      }

      cur_weight += u_weight;
      _piercing_nodes.push_back(u);

      if (_piercing_nodes.size() >= max_num_piercing_nodes) {
        return;
      }
    }
  };

  const std::size_t max_num_piercing_nodes =
      _ctx.bulk_piercing
          ? _bulk_piercing_ctx[terminal_set].compute_max_num_piercing_nodes(terminal_set_weight)
          : 1;
  add_piercing_nodes(
      _unreachable_piercing_node_candidates, std::numeric_limits<std::size_t>::max()
  );
  add_piercing_nodes(_reachable_piercing_node_candidates, max_num_piercing_nodes);

  if (_ctx.bulk_piercing) {
    _bulk_piercing_ctx[terminal_set].register_num_pierced_nodes(_piercing_nodes.size());
  }
}

} // namespace kaminpar::shm
