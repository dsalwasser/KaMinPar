#include "kaminpar-shm/refinement/flow/piercing/random_piercing_heuristic.h"

#include <cstddef>

#include "kaminpar-common/random.h"

namespace kaminpar::shm {

RandomPiercingHeuristic::RandomPiercingHeuristic(const MultiwayPiercingHeuristicContext &ctx)
    : _ctx(ctx) {}

void RandomPiercingHeuristic::initialize(
    const CSRGraph &graph,
    [[maybe_unused]] const PartitionedCSRGraph &p_graph,
    const PartitionContext &p_ctx,
    const TerminalSets &terminal_sets
) {
  _graph = &graph;
  _terminal_sets = &terminal_sets;

  _piercing_nodes_candidates_marker.resize(graph.n());

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

std::span<const NodeID> RandomPiercingHeuristic::find_piercing_nodes(
    const BlockID terminal_set, const NodeWeight terminal_set_weight, const NodeWeight max_weight
) {
  _piercing_nodes_candidates.clear();
  _piercing_nodes_candidates_marker.reset();

  for (const NodeID u : _terminal_sets->terminal_set_nodes(terminal_set)) {
    _graph->adjacent_nodes(u, [&](const NodeID v) {
      if (_terminal_sets->is_terminal(v) || _piercing_nodes_candidates_marker.get(v)) {
        return;
      }

      const NodeWeight v_weight = _graph->node_weight(v);
      if (v_weight > max_weight) {
        return;
      }

      _piercing_nodes_candidates_marker.set(v);
      _piercing_nodes_candidates.push_back(v);
    });
  }

  const std::size_t max_num_piercing_nodes =
      _ctx.bulk_piercing
          ? _bulk_piercing_ctx[terminal_set].compute_max_num_piercing_nodes(terminal_set_weight)
          : 1;
  const std::size_t num_piercing_nodes =
      std::min(_piercing_nodes_candidates.size(), max_num_piercing_nodes);

  Random &random = Random::instance();
  for (std::size_t i = 0; i < num_piercing_nodes; ++i) {
    const std::size_t index = random.random_index(i, _piercing_nodes_candidates.size());
    std::swap(_piercing_nodes_candidates[i], _piercing_nodes_candidates[index]);
  }

  if (_ctx.bulk_piercing) {
    _bulk_piercing_ctx[terminal_set].register_num_pierced_nodes(num_piercing_nodes);
  }

  return std::span(_piercing_nodes_candidates.data(), num_piercing_nodes);
}

} // namespace kaminpar::shm
