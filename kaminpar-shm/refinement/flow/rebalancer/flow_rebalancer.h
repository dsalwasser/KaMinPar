#pragma once

#include <algorithm>
#include <cstdint>
#include <span>

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/datastructures/delta_partitioned_graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/flow/flow_network/border_region.h"
#include "kaminpar-shm/refinement/flow/flow_network/flow_network.h"
#include "kaminpar-shm/refinement/gains/delta_gain_caches.h"

#include "kaminpar-common/datastructures/binary_heap.h"
#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm {

class FlowRebalancer {
public:
  struct Move {
    NodeID node;
    BlockID target_block;
  };

  struct Result {
    bool balanced;
    BlockID overloaded_block;

    EdgeWeight gain;
    std::span<const Move> moves;

    [[nodiscard]] static Result success(
        const BlockID overloaded_block, const EdgeWeight gain, const std::span<const Move> moves
    ) {
      return Result(true, overloaded_block, gain, moves);
    }

    [[nodiscard]] static Result failure() {
      return Result(false, kInvalidBlockID, 0, {});
    }
  };

  virtual ~FlowRebalancer() = default;

  virtual void initialize(
      bool source_side_cut, const BorderRegion &border_region, const FlowNetwork &flow_network
  ) = 0;

  virtual void update_nodes(std::span<const NodeID> nodes) = 0;

  virtual Result rebalance() = 0;

  virtual void revert_moves() = 0;

  virtual const DeltaPartitionedCSRGraph &d_graph() const = 0;
};

class FlowRebalancerMoves {
public:
  using Move = FlowRebalancer::Move;

  static constexpr std::uint8_t kUninitialized = 0;
  static constexpr std::uint8_t kPendingInitialization = 1;
  static constexpr std::uint8_t kInitialized = 2;

  void initialize(const BlockID k) {
    _initialized_moves.resize(k, static_array::noinit);
    std::fill_n(_initialized_moves.begin(), k, kUninitialized);

    _precomputed_moves.resize(k);
  }

  std::uint8_t is_initialized(const BlockID block) {
    std::uint8_t expected = kUninitialized;
    if (__atomic_compare_exchange_n(
            &_initialized_moves[block],
            &expected,
            kPendingInitialization,
            false,
            __ATOMIC_ACQ_REL,
            __ATOMIC_ACQUIRE
        )) {
      return kUninitialized;
    }

    return __atomic_load_n(&_initialized_moves[block], __ATOMIC_ACQUIRE);
  }

  void set_initialized(const BlockID block) {
    __atomic_store_n(&_initialized_moves[block], kInitialized, __ATOMIC_RELAXED);
  }

  ScalableVector<Move> &moves(const BlockID block) {
    return _precomputed_moves[block];
  }

private:
  StaticArray<std::uint8_t> _initialized_moves;
  ScalableVector<ScalableVector<Move>> _precomputed_moves;
};

template <typename GainCache> class FlowRebalancerBase : public FlowRebalancer {
  using RelativeGain = float;
  using PriorityQueue = BinaryMaxHeap<RelativeGain>;

public:
  using DeltaGainCache = GenericDeltaGainCache<GainCache>;

  FlowRebalancerBase(
      const PartitionedCSRGraph &p_graph,
      const GainCache &gain_cache,
      std::span<const BlockWeight> max_block_weights
  )
      : _graph(p_graph.graph()),
        _max_block_weights(max_block_weights),
        _d_graph(&p_graph),
        _gain_cache(gain_cache, _d_graph),
        _priority_queue(_graph.n()),
        _target_blocks(_graph.n(), static_array::noinit) {}

  [[nodiscard]] bool has_next_node() const {
    return !_priority_queue.empty();
  }

  std::pair<NodeID, BlockID> next_node() {
    const NodeID u = _priority_queue.peek_id();
    _priority_queue.pop();

    const BlockID target_block = _target_blocks[u];
    return {u, target_block};
  }

  void insert_node(const NodeID u) {
    const auto [target_block, relative_gain] = compute_best_move(u);
    if (target_block == kInvalidBlockID) {
      return;
    }

    _priority_queue.push(u, relative_gain);
    _target_blocks[u] = target_block;
  }

  EdgeWeight move_node(const NodeID u, const BlockID source_block, const BlockID target_block) {
    const EdgeWeight gain = _gain_cache.gain(u, source_block, target_block);
    _gain_cache.move(u, source_block, target_block);

    _d_graph.set_block(u, target_block);
    _graph.adjacent_nodes(u, [&](const NodeID v) { update_node(v); });

    return gain;
  }

  void clear_nodes() {
    _priority_queue.clear();
  }

private:
  void update_node(const NodeID u) {
    if (!_priority_queue.contains(u)) {
      return;
    }

    const auto [target_block, relative_gain] = compute_best_move(u);
    if (target_block == kInvalidBlockID) {
      _priority_queue.remove(u);
      return;
    }

    _priority_queue.change_priority(u, relative_gain);
    _target_blocks[u] = target_block;
  }

  [[nodiscard]] std::pair<BlockID, RelativeGain> compute_best_move(const NodeID u) const {
    const BlockID u_block = _d_graph.block(u);
    const NodeWeight u_weight = _graph.node_weight(u);

    BlockID target_block = kInvalidBlockID;
    BlockWeight target_block_weight = std::numeric_limits<BlockWeight>::max();
    EdgeWeight target_block_connection = std::numeric_limits<RelativeGain>::min();

    for (BlockID block = 0, k = _d_graph.k(); block < k; ++block) {
      if (block == u_block) {
        continue;
      }

      const BlockWeight block_weight = _d_graph.block_weight(block);
      if (block_weight + u_weight > _max_block_weights[block]) {
        continue;
      }

      const EdgeWeight block_connection = _gain_cache.conn(u, block);
      if (block_connection > target_block_connection ||
          (block_connection == target_block_connection && block_weight < target_block_weight)) {
        target_block = block;
        target_block_weight = block_weight;
        target_block_connection = block_connection;
      }
    }

    if (target_block == kInvalidBlockID) {
      return {kInvalidBlockID, 0};
    }

    const EdgeWeight from_connection = _gain_cache.conn(u, u_block);
    const EdgeWeight absolute_gain = target_block_connection - from_connection;

    const RelativeGain relative_gain = compute_relative_gain(absolute_gain, u_weight);
    return {target_block, relative_gain};
  }

  [[nodiscard]] static RelativeGain
  compute_relative_gain(const EdgeWeight absolute_gain, const NodeWeight weight) {
    return (absolute_gain >= 0) ? (absolute_gain * weight)
                                : (absolute_gain / static_cast<RelativeGain>(weight));
  }

protected:
  const CSRGraph &_graph;
  std::span<const BlockWeight> _max_block_weights;

  DeltaPartitionedCSRGraph _d_graph;
  DeltaGainCache _gain_cache;

private:
  PriorityQueue _priority_queue;
  StaticArray<BlockID> _target_blocks;
};

} // namespace kaminpar::shm
