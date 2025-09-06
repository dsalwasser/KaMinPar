#pragma once

#include "kaminpar-shm/refinement/flow/flow_network/border_region.h"
#include "kaminpar-shm/refinement/flow/flow_network/flow_network.h"
#include "kaminpar-shm/refinement/flow/rebalancer/flow_rebalancer.h"

#include "kaminpar-common/datastructures/scalable_vector.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

template <typename Graph, typename PartitionedGraph, typename GainCache>
class StaticFlowRebalancer : public FlowRebalancerBase<Graph, PartitionedGraph, GainCache> {
  using Base = FlowRebalancerBase<Graph, PartitionedGraph, GainCache>;

  using Base::_gain_cache;
  using Base::_graph;
  using Base::_max_block_weights;
  using Base::_p_graph;

public:
  using Move = Base::Move;
  using Result = Base::Result;

  StaticFlowRebalancer(
      PartitionedGraph &p_graph,
      GainCache &gain_cache,
      std::span<const BlockWeight> max_block_weights
  )
      : Base(p_graph, gain_cache, max_block_weights),
        _overloaded_block(kInvalidBlockID) {}

  void initialize(const BorderRegion &border_region, const FlowNetwork &flow_network) override {
    _border_region = &border_region;
    _flow_network = &flow_network;

    _initialized_source_side_moves = false;
    _initialized_sink_side_moves = false;
  }

  [[nodiscard]] Result rebalance(const BlockID overloaded_block) override {
    KASSERT(_moves.empty());
    KASSERT(
        overloaded_block == _border_region->block1() || overloaded_block == _border_region->block2()
    );

    const bool balance_source_side = overloaded_block == _border_region->block1();
    if (balance_source_side) {
      if (!_initialized_source_side_moves) {
        _initialized_source_side_moves = true;
        compute_move_order(_border_region->block1(), _source_side_moves);
      }
    } else {
      if (!_initialized_sink_side_moves) {
        _initialized_sink_side_moves = true;
        compute_move_order(_border_region->block2(), _sink_side_moves);
      }
    }

    _overloaded_block = overloaded_block;
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
  void compute_move_order(const BlockID overloaded_block, ScalableVector<Move> &moves) {
    SCOPED_TIMER("Compute Move Order");

    TIMED_SCOPE("Update State") {
      _virtual_moves.clear();

      for (const auto &[u, _] : _flow_network->global_to_local_mapping.entries()) {
        const BlockID u_block = _p_graph.block(u);
        if (u_block == overloaded_block) {
          continue;
        }

        _p_graph.set_block(u, overloaded_block);
        _gain_cache.move(u, u_block, overloaded_block);

        _virtual_moves.emplace_back(u, u_block);
      }
    };

    TIMED_SCOPE("Insert Nodes") {
      Base::clear_nodes();

      for (const NodeID u : _graph.nodes()) {
        if (_p_graph.block(u) == overloaded_block) {
          Base::insert_node(u);
        }
      }
    };

    TIMED_SCOPE("Compute Moves") {
      moves.clear();

      while (Base::has_next_node()) {
        const auto [u, target_block] = Base::next_node();

        if (_p_graph.block_weight(target_block) + _graph.node_weight(u) >
            _max_block_weights[target_block]) {
          Base::insert_node(u);
          continue;
        }

        Base::move_node(u, overloaded_block, target_block);
        moves.emplace_back(u, target_block);
      }
    };

    TIMED_SCOPE("Revert Moves") {
      for (const auto &[u, target_block] : moves) {
        _p_graph.set_block(u, overloaded_block);
        _gain_cache.move(u, target_block, overloaded_block);
      }

      for (const auto &[u, u_block] : _virtual_moves) {
        _p_graph.set_block(u, u_block);
        _gain_cache.move(u, overloaded_block, u_block);
      }
    };
  }

  Result compute_moves() {
    SCOPED_TIMER("Compute Moves");

    const BlockID overloaded_block = _overloaded_block;
    const BlockWeight max_block_weight = _max_block_weights[overloaded_block];

    const bool balance_source_side = overloaded_block == _border_region->block1();
    const std::span<const Move> move_order =
        balance_source_side ? _source_side_moves : _sink_side_moves;

    NodeID cur_move = 0;
    EdgeWeight gain = 0;
    while (_p_graph.block_weight(overloaded_block) > max_block_weight) {
      while (true) {
        if (cur_move >= move_order.size()) {
          return Result::failure();
        }

        const auto [u, target_block] = move_order[cur_move++];
        const BlockID u_block = _p_graph.block(u);

        if (u_block != overloaded_block) {
          continue;
        }

        if (_p_graph.block_weight(target_block) + _graph.node_weight(u) >
            _max_block_weights[target_block]) {
          continue;
        }

        _p_graph.set_block(u, target_block);
        _moves.emplace_back(u, target_block);

        gain += _gain_cache.gain(u, u_block, target_block);
        _gain_cache.move(u, u_block, target_block);

        break;
      }
    }

    return Result(true, gain, _moves);
  }

private:
  const BorderRegion *_border_region;
  const FlowNetwork *_flow_network;

  bool _initialized_source_side_moves;
  bool _initialized_sink_side_moves;
  ScalableVector<Move> _virtual_moves;

  ScalableVector<Move> _source_side_moves;
  ScalableVector<Move> _sink_side_moves;

  BlockID _overloaded_block;
  ScalableVector<Move> _moves;
};

} // namespace kaminpar::shm
