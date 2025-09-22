#pragma once

#include <span>

#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/flow/flow_network/border_region.h"
#include "kaminpar-shm/refinement/flow/flow_network/flow_network.h"
#include "kaminpar-shm/refinement/flow/rebalancer/flow_rebalancer.h"

#include "kaminpar-common/datastructures/scalable_vector.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

template <typename GainCache> class StaticFlowRebalancer : public FlowRebalancerBase<GainCache> {
  using Base = FlowRebalancerBase<GainCache>;

  using Base::_d_graph;
  using Base::_gain_cache;
  using Base::_graph;
  using Base::_max_block_weights;

public:
  using Move = Base::Move;
  using Result = Base::Result;
  using DeltaGainCache = Base::DeltaGainCache;

  StaticFlowRebalancer(
      const PartitionedCSRGraph &p_graph,
      const GainCache &gain_cache,
      const std::span<const BlockWeight> max_block_weights
  )
      : Base(p_graph, gain_cache, max_block_weights) {}

  void initialize(
      const bool source_side_cut, const BorderRegion &border_region, const FlowNetwork &flow_network
  ) override {
    SCOPED_TIMER("Initialize Rebalancer");

    _source_side_cut = source_side_cut;
    _flow_network = &flow_network;

    _block1 = border_region.block1();
    _block2 = border_region.block2();

    _initialized_source_side_moves = false;
    _initialized_sink_side_moves = false;

    initialize_state();
  }

  void update_nodes(const std::span<const NodeID> nodes) override {
    SCOPED_TIMER("Update Node State");

    const BlockID target_block = _source_side_cut ? _block1 : _block2;
    for (const NodeID u_local : nodes) {
      if (u_local == FlowNetwork::source || u_local == FlowNetwork::sink) {
        continue;
      }

      const NodeID u = _flow_network->local_to_global_mapping.get(u_local);
      const BlockID u_block = _d_graph.block(u);

      if (u_block != target_block) {
        _d_graph.set_block(u, target_block);
        _gain_cache.move(u, u_block, target_block);
      }
    }
  }

  [[nodiscard]] Result rebalance() override {
    KASSERT(_moves.empty());

    _source_side_moves = _d_graph.block_weight(_block1) > _max_block_weights[_block1];
    const BlockID overloaded_block = _source_side_moves ? _block1 : _block2;

    if (_source_side_moves) {
      if (!_initialized_source_side_moves) {
        _initialized_source_side_moves = true;
        compute_move_order(_block1, _precomputed_source_side_moves);
      }
    } else {
      if (!_initialized_sink_side_moves) {
        _initialized_sink_side_moves = true;
        compute_move_order(_block2, _precomputed_sink_side_moves);
      }
    }

    const std::span<const Move> move_order =
        _source_side_moves ? _precomputed_source_side_moves : _precomputed_sink_side_moves;
    const BlockWeight max_block_weight = _max_block_weights[overloaded_block];

    return TIMED_SCOPE("Extract Nodes") {
      NodeID cur_move = 0;
      EdgeWeight gain = 0;

      while (_d_graph.block_weight(overloaded_block) > max_block_weight) {
        while (true) {
          if (cur_move == move_order.size()) {
            return Result::failure();
          }

          const auto [u, target_block] = move_order[cur_move++];
          const BlockID u_block = _d_graph.block(u);

          if (u_block != overloaded_block) {
            continue;
          }

          if (_d_graph.block_weight(target_block) + _graph.node_weight(u) >
              _max_block_weights[target_block]) {
            continue;
          }

          gain += _gain_cache.gain(u, u_block, target_block);
          _gain_cache.move(u, u_block, target_block);

          _d_graph.set_block(u, target_block);
          _moves.emplace_back(u, target_block);
          break;
        }
      }

      return Result::success(overloaded_block, gain, _moves);
    };
  }

  void revert_moves() override {
    SCOPED_TIMER("Revert Moves");

    const BlockID overloaded_block = _source_side_moves ? _block1 : _block2;
    for (const auto &[u, target_block] : _moves) {
      KASSERT(_d_graph.block(u) == target_block);

      _d_graph.set_block(u, overloaded_block);
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
        const BlockID u_block = _d_graph.block(u);
        if (u_block == overloaded_block) {
          continue;
        }

        _d_graph.set_block(u, overloaded_block);
        _gain_cache.move(u, u_block, overloaded_block);

        _virtual_moves.emplace_back(u, u_block);
      }
    };

    TIMED_SCOPE("Insert Nodes") {
      Base::clear_nodes();

      for (const NodeID u : _graph.nodes()) {
        if (_d_graph.block(u) == overloaded_block) {
          Base::insert_node(u);
        }
      }
    };

    TIMED_SCOPE("Extract Nodes") {
      moves.clear();

      while (Base::has_next_node()) {
        const auto [u, target_block] = Base::next_node();

        if (_d_graph.block_weight(target_block) + _graph.node_weight(u) >
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
        _d_graph.set_block(u, overloaded_block);
        _gain_cache.move(u, target_block, overloaded_block);
      }

      for (const auto &[u, u_block] : _virtual_moves) {
        _d_graph.set_block(u, u_block);
        _gain_cache.move(u, overloaded_block, u_block);
      }
    };
  }

  [[nodiscard]] const DeltaPartitionedCSRGraph &d_graph() const override {
    return _d_graph;
  }

private:
  void initialize_state() {
    SCOPED_TIMER("Initialize State");

    _d_graph.clear();
    _gain_cache.clear();

    const BlockID target_block = _source_side_cut ? _block2 : _block1;
    for (const auto &[u, _] : _flow_network->global_to_local_mapping.entries()) {
      const BlockID u_block = _d_graph.block(u);

      if (u_block != target_block) {
        _d_graph.set_block(u, target_block);
        _gain_cache.move(u, u_block, target_block);
      }
    }
  }

private:
  bool _source_side_cut;
  const FlowNetwork *_flow_network;

  BlockID _block1;
  BlockID _block2;

  bool _initialized_source_side_moves;
  bool _initialized_sink_side_moves;
  ScalableVector<Move> _virtual_moves;

  ScalableVector<Move> _precomputed_source_side_moves;
  ScalableVector<Move> _precomputed_sink_side_moves;

  bool _source_side_moves;
  ScalableVector<Move> _moves;
};

} // namespace kaminpar::shm
