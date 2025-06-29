#pragma once

#include <algorithm>
#include <cstddef>
#include <span>

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/flow/util/terminal_sets.h"

namespace kaminpar::shm {

class BulkPiercingContext {
public:
  BulkPiercingContext(
      std::size_t bulk_piercing_round_threshold, double bulk_piercing_shrinking_factor
  )
      : _bulk_piercing_round_threshold(bulk_piercing_round_threshold),
        _bulk_piercing_shrinking_factor(bulk_piercing_shrinking_factor) {}

  void initialize(
      const NodeWeight side_weight,
      const NodeWeight total_weight,
      const NodeWeight max_side_weight,
      const NodeWeight max_total_weight
  ) {
    _num_rounds = 0;
    _total_bulk_piercing_nodes = 0;

    _initial_side_weight = side_weight;
    _weight_added_so_far = 0;

    const double ratio = max_side_weight / static_cast<double>(max_total_weight);
    _current_weight_goal = std::max(0.0, ratio * total_weight - side_weight);
    _current_weight_goal_remaining = 0;
  }

  void register_num_pierced_nodes(const NodeID num_pierced_nodes) {
    _total_bulk_piercing_nodes += num_pierced_nodes;
  }

  std::size_t compute_max_num_piercing_nodes(const NodeWeight side_weight) {
    if (++_num_rounds <= _bulk_piercing_round_threshold) {
      return 1;
    }

    _current_weight_goal *= _bulk_piercing_shrinking_factor;
    _current_weight_goal_remaining += _current_weight_goal;

    const NodeWeight added_weight = side_weight - (_initial_side_weight + _weight_added_so_far);
    _weight_added_so_far += added_weight;
    _current_weight_goal_remaining -= added_weight;

    const double speed = _weight_added_so_far / static_cast<double>(_total_bulk_piercing_nodes);
    if (_current_weight_goal_remaining <= speed) {
      return 1;
    }

    const std::size_t estimated_num_piercing_nodes = _current_weight_goal_remaining / speed;
    return estimated_num_piercing_nodes;
  }

private:
  std::size_t _bulk_piercing_round_threshold;
  double _bulk_piercing_shrinking_factor;

  std::size_t _num_rounds;
  std::size_t _total_bulk_piercing_nodes;

  NodeWeight _initial_side_weight;
  NodeWeight _weight_added_so_far;
  NodeWeight _current_weight_goal;
  NodeWeight _current_weight_goal_remaining;
};

class MultiwayPiercingHeuristic {
public:
  virtual ~MultiwayPiercingHeuristic() = default;

  virtual void initialize(
      const CSRGraph &graph,
      const PartitionedCSRGraph &p_graph,
      const PartitionContext &p_ctx,
      const TerminalSets &terminal_sets
  ) = 0;

  virtual std::span<const NodeID> find_piercing_nodes(
      BlockID terminal_set_to_pierce, const NodeWeight terminal_set_weight, NodeWeight max_weight
  ) = 0;
};

} // namespace kaminpar::shm
