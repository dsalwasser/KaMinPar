#pragma once

#include <span>
#include <unordered_set>

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/flow/util/node_status.h"

#include "kaminpar-common/datastructures/binary_heap.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/logger.h"

namespace kaminpar::shm {

class PiercingHeuristic {
  SET_DEBUG(false);

  class PiercingNodeCandidatesBuckets {
  public:
    void initialize(const NodeID max_distance) {
      _candidates_buckets.resize(max_distance + 1);
    }

    void reset() {
      for (ScalableVector<NodeID> &candidates : _candidates_buckets) {
        candidates.clear();
      }
    }

    void add_candidate(const NodeID u, const NodeID distance) {
      KASSERT(distance < _candidates_buckets.size());
      _candidates_buckets[distance].push_back(u);
    }

    [[nodiscard]] NodeID min_occupied_bucket() const {
      return 0;
    }

    [[nodiscard]] NodeID max_occupied_bucket() const {
      return _candidates_buckets.size() - 1;
    }

    [[nodiscard]] std::span<const NodeID> candidates(const NodeID bucket) const {
      KASSERT(bucket < _candidates_buckets.size());

      return _candidates_buckets[bucket];
    }

  private:
    ScalableVector<ScalableVector<NodeID>> _candidates_buckets;
  };

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

    void register_num_nodes(const NodeID num_pierced_nodes) {
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

public:
  PiercingHeuristic(const PiercingHeuristicContext &_ctx);

  void initialize(
      const CSRGraph &graph,
      const std::unordered_set<NodeID> &initial_source_side_nodes,
      const std::unordered_set<NodeID> &initial_sink_side_nodes,
      NodeWeight source_side_weight,
      NodeWeight sink_side_weight,
      NodeWeight total_weight,
      NodeWeight max_source_side_weight,
      NodeWeight max_sink_side_weight
  );

  void reset(bool source_side);

  void add_piercing_node_candidate(NodeID node, bool reachable);

  std::span<const NodeID> find_piercing_nodes(
      const NodeStatus &cut_status,
      const NodeStatus &terminal_status,
      NodeWeight side_weight,
      NodeWeight max_weight
  );

private:
  NodeID compute_distances();

  std::size_t compute_max_num_piercing_nodes(NodeWeight side_weight);

  BulkPiercingContext &bulk_piercing_context();

private:
  const PiercingHeuristicContext &_ctx;

  const CSRGraph *_graph;
  const std::unordered_set<NodeID> *_initial_source_side_nodes;
  const std::unordered_set<NodeID> *_initial_sink_side_nodes;

  ScalableVector<NodeID> _piercing_nodes;
  StaticArray<NodeID> _distance;

  bool _source_side;
  const std::unordered_set<NodeID> *_initial_side_nodes;
  PiercingNodeCandidatesBuckets _reachable_candidates_buckets;
  PiercingNodeCandidatesBuckets _unreachable_candidates_buckets;

  BulkPiercingContext _source_side_bulk_piercing_ctx;
  BulkPiercingContext _sink_side_bulk_piercing_ctx;
};

} // namespace kaminpar::shm
