#pragma once

#include <span>

#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/flow/rebalancer/flow_rebalancer.h"

#include "kaminpar-common/datastructures/scalable_vector.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

template <typename Graph, typename PartitionedGraph, typename GainCache>
class DynamicFlowRebalancer : public FlowRebalancerBase<Graph, PartitionedGraph, GainCache> {
  using Base = FlowRebalancerBase<Graph, PartitionedGraph, GainCache>;

  using Base::_gain_cache;
  using Base::_graph;
  using Base::_max_block_weights;
  using Base::_p_graph;

public:
  using Move = Base::Move;
  using Result = Base::Result;

  DynamicFlowRebalancer(
      PartitionedGraph &p_graph,
      GainCache &gain_cache,
      std::span<const BlockWeight> max_block_weights
  )
      : Base(p_graph, gain_cache, max_block_weights),
        _overloaded_block(kInvalidBlockID) {}

  void initialize(
      [[maybe_unused]] const BorderRegion &border_region,
      [[maybe_unused]] const FlowNetwork &flow_network
  ) override {}

  [[nodiscard]] Result rebalance(const BlockID overloaded_block) override {
    KASSERT(_moves.empty());
    _overloaded_block = overloaded_block;

    insert_nodes();
    return compute_moves();
  }

  void revert_moves() override {
    SCOPED_TIMER("Revert Moves");

    const BlockID overloaded_block = _overloaded_block;
    for (const auto &[u, target_block] : _moves) {
      _p_graph.set_block(u, overloaded_block);
      _gain_cache.move(u, target_block, overloaded_block);
    }

    _moves.clear();
  }

private:
  void insert_nodes() {
    SCOPED_TIMER("Insert Nodes");

    Base::clear_nodes();

    const BlockID overloaded_block = _overloaded_block;
    for (const NodeID u : _graph.nodes()) {
      if (_p_graph.block(u) == overloaded_block) {
        Base::insert_node(u);
      }
    }
  }

  Result compute_moves() {
    SCOPED_TIMER("Compute Moves");

    const BlockID overloaded_block = _overloaded_block;
    const BlockWeight max_block_weight = _max_block_weights[overloaded_block];

    EdgeWeight gain = 0;
    while (_p_graph.block_weight(overloaded_block) > max_block_weight) {
      while (true) {
        if (!Base::has_next_node()) {
          return Result::failure();
        }

        const auto [u, target_block] = Base::next_node();
        if (_p_graph.block_weight(target_block) + _graph.node_weight(u) >
            _max_block_weights[target_block]) {
          Base::insert_node(u);
          continue;
        }

        gain += Base::move_node(u, overloaded_block, target_block);
        _moves.emplace_back(u, target_block);

        break;
      }
    }

    return Result(true, gain, _moves);
  }

private:
  BlockID _overloaded_block;
  ScalableVector<Move> _moves;
};

} // namespace kaminpar::shm
