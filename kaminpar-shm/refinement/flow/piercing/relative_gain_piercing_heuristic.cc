#include "kaminpar-shm/refinement/flow/piercing/relative_gain_piercing_heuristic.h"

namespace kaminpar::shm {

RelativeGainPiercingHeuristic::RelativeGainPiercingHeuristic(
    const MultiwayPiercingHeuristicContext &ctx
)
    : _ctx(ctx) {}

void RelativeGainPiercingHeuristic::initialize(
    const CSRGraph &graph,
    const PartitionedCSRGraph &p_graph,
    const PartitionContext &p_ctx,
    const TerminalSets &terminal_sets
) {
  _graph = &graph;
  _p_graph = &p_graph;
  _terminal_sets = &terminal_sets;

  if (_priority_queue.size() < graph.n()) {
    _priority_queue.resize(graph.n());
  }

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

std::span<const NodeID> RelativeGainPiercingHeuristic::find_piercing_nodes(
    const BlockID terminal_set, const NodeWeight terminal_set_weight, const NodeWeight max_weight
) {
  _piercing_nodes.clear();
  _priority_queue.clear();

  for (const NodeID terminal : _terminal_sets->terminal_set_nodes(terminal_set)) {
    _graph->adjacent_nodes(terminal, [&](const NodeID u) {
      if (_terminal_sets->is_terminal(u) || _priority_queue.contains(u)) {
        return;
      }

      const NodeWeight u_weight = _graph->node_weight(u);
      if (u_weight > max_weight) {
        return;
      }

      const BlockID u_block = _p_graph->block(u);

      EdgeWeight from_connection = 0;
      EdgeWeight to_connection = 0;
      _graph->adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
        const BlockID v_block = _p_graph->block(v);
        from_connection += (v_block == u_block) ? w : 0;
        to_connection += (v_block == terminal_set) ? w : 0;
      });

      const EdgeWeight absolute_gain = to_connection - from_connection;
      const RelativeGain relative_gain = (absolute_gain >= 0)
                                             ? (absolute_gain * u_weight)
                                             : (absolute_gain / static_cast<float>(u_weight));
      _priority_queue.push(u, relative_gain);
    });
  }

  const std::size_t max_num_piercing_nodes =
      _ctx.bulk_piercing
          ? _bulk_piercing_ctx[terminal_set].compute_max_num_piercing_nodes(terminal_set_weight)
          : 1;
  const std::size_t num_piercing_nodes = std::min(_priority_queue.size(), max_num_piercing_nodes);

  for (std::size_t i = 0; i < num_piercing_nodes; ++i) {
    const NodeID piercing_node = _priority_queue.peek_id();
    _priority_queue.pop();

    _piercing_nodes.push_back(piercing_node);
  }

  if (_ctx.bulk_piercing) {
    _bulk_piercing_ctx[terminal_set].register_num_pierced_nodes(num_piercing_nodes);
  }

  return _piercing_nodes;
}

} // namespace kaminpar::shm
