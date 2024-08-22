/*******************************************************************************
 * Wrapper graph for a block induced subgraph.
 *
 * @file:   block_induced_subgraph.h
 * @author: Daniel Salwasser
 * @date:   19.08.2024
 ******************************************************************************/
#pragma once

#include <utility>

#include "kaminpar-shm/datastructures/abstract_graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/ranges.h"

namespace kaminpar::shm {

class LocalToGlobalMapping {
public:
  LocalToGlobalMapping(
      const bool use_shared,
      const NodeID offset,
      NodeID *shared_local_to_global,
      NodeID *backup_local_to_global
  )
      : _use_shared(use_shared),
        _offset(offset),
        _local_to_global((use_shared ? shared_local_to_global : backup_local_to_global) + offset),
        _shared_local_to_global(shared_local_to_global),
        _backup_local_to_global(backup_local_to_global) {}

  [[nodiscard]] bool use_shared() const {
    return _use_shared;
  }

  [[nodiscard]] std::size_t offset() const {
    return _offset;
  }

  [[nodiscard]] const NodeID &operator[](const NodeID u_local) const {
    return _local_to_global[u_local];
  }

  NodeID &operator[](const NodeID u_local) {
    return _local_to_global[u_local];
  }

  [[nodiscard]] NodeID *shared() const {
    return _shared_local_to_global;
  }

  [[nodiscard]] NodeID *backup() const {
    return _backup_local_to_global;
  }

private:
  bool _use_shared;
  NodeID _offset;
  NodeID *_local_to_global;
  NodeID *_shared_local_to_global;
  NodeID *_backup_local_to_global;
};

template <typename Graph> class BlockInducedSubgraph : public AbstractGraph {
public:
  using NodeID = Graph::NodeID;
  using NodeWeight = Graph::NodeWeight;
  using EdgeID = Graph::EdgeID;
  using EdgeWeight = Graph::EdgeWeight;

  BlockInducedSubgraph(
      const Graph *graph,
      StaticArray<BlockID> partition,
      const NodeID num_nodes,
      const EdgeID num_edges,
      const NodeWeight max_node_weight,
      const NodeWeight total_node_weight,
      const EdgeWeight total_edge_weight,
      const BlockID block,
      StaticArray<NodeID> global_to_local,
      LocalToGlobalMapping local_to_global
  )
      : _graph(graph),
        _partition(std::move(partition)),
        _num_nodes(num_nodes),
        _num_edges(num_edges),
        _max_node_weight(max_node_weight),
        _total_node_weight(total_node_weight),
        _total_edge_weight(total_edge_weight),
        _block(block),
        _global_to_local(std::move(global_to_local)),
        _local_to_global(std::move(local_to_global)) {
    KASSERT(partition.is_span());
    KASSERT(global_to_local.is_span());
  }

  ~BlockInducedSubgraph() = default;

  BlockInducedSubgraph(const BlockInducedSubgraph &) = delete;
  BlockInducedSubgraph &operator=(const BlockInducedSubgraph &) = delete;

  BlockInducedSubgraph(BlockInducedSubgraph &&) noexcept = default;
  BlockInducedSubgraph &operator=(BlockInducedSubgraph &&) noexcept = default;

  [[nodiscard]] inline StaticArray<EdgeID> &raw_nodes() {
    __builtin_unreachable();
  }

  void remove_isolated_nodes(const NodeID num_isolated_nodes) {
    __builtin_unreachable();
  }

  void integrate_isolated_nodes() {
    __builtin_unreachable();
  }

  //
  // Size of the graph
  //

  [[nodiscard]] inline NodeID n() const final {
    return _num_nodes;
  }

  [[nodiscard]] inline EdgeID m() const final {
    return _num_edges;
  }

  //
  // Node and edge weights
  //

  [[nodiscard]] inline bool is_node_weighted() const final {
    return _graph->is_node_weighted();
  }

  [[nodiscard]] inline NodeWeight node_weight(const NodeID u) const final {
    return _graph->node_weight(_local_to_global[u]);
  }

  [[nodiscard]] inline NodeWeight max_node_weight() const final {
    return _max_node_weight;
  }

  [[nodiscard]] inline NodeWeight total_node_weight() const final {
    return _total_node_weight;
  }

  [[nodiscard]] inline StaticArray<NodeWeight> &raw_node_weights() const {
    __builtin_unreachable();
  }

  inline void update_total_node_weight() final {
    __builtin_unreachable();
  }

  [[nodiscard]] inline bool is_edge_weighted() const final {
    return _graph->is_edge_weighted();
  }

  [[nodiscard]] inline EdgeWeight total_edge_weight() const final {
    return _total_edge_weight;
  }

  //
  // Iterators for nodes / edges
  //

  [[nodiscard]] inline IotaRange<NodeID> nodes() const final {
    return {static_cast<NodeID>(0), _num_nodes};
  }

  [[nodiscard]] inline IotaRange<EdgeID> edges() const final {
    return {static_cast<NodeID>(0), _num_edges};
  }

  [[nodiscard]] inline IotaRange<EdgeID> incident_edges(const NodeID u) const final {
    __builtin_unreachable();
  }

  //
  // Node degree
  //

  [[nodiscard]] inline NodeID max_degree() const final {
    __builtin_unreachable();
  }

  [[nodiscard]] inline NodeID degree(const NodeID u) const final {
    return _graph->degree(_local_to_global[u]);
  }

  //
  // Graph operations
  //

  template <typename Lambda> inline void adjacent_nodes(const NodeID u, Lambda &&l) const {
    constexpr bool kDontDecodeEdgeWeights = std::is_invocable_v<Lambda, NodeID>;
    constexpr bool kDecodeEdgeWeights = std::is_invocable_v<Lambda, NodeID, EdgeWeight>;
    static_assert(kDontDecodeEdgeWeights || kDecodeEdgeWeights);

    using LambdaReturnType = std::conditional_t<
        kDecodeEdgeWeights,
        std::invoke_result<Lambda, NodeID, EdgeWeight>,
        std::invoke_result<Lambda, NodeID>>::type;
    constexpr bool kNonStoppable = std::is_void_v<LambdaReturnType>;

    const NodeID u_global = _local_to_global[u];
    if constexpr (kDecodeEdgeWeights) {
      _graph->adjacent_nodes(u_global, [&](const NodeID v, const EdgeWeight w) {
        const BlockID v_block = _partition[v];
        if (v_block != _block) {
          if constexpr (kNonStoppable) {
            return;
          } else {
            return false;
          }
        }

        const NodeID v_local = _global_to_local[v];
        return l(v_local, w);
      });
    } else {
      _graph->adjacent_nodes(u_global, [&](const NodeID v) {
        const BlockID v_block = _partition[v];
        if (v_block != _block) {
          if constexpr (kNonStoppable) {
            return;
          } else {
            return false;
          }
        }

        const NodeID v_local = _global_to_local[v];
        return l(v_local);
      });
    }
  }

  template <typename Lambda> inline void neighbors(const NodeID u, Lambda &&l) const {
    __builtin_unreachable();
  }

  template <typename Lambda>
  inline void neighbors(const NodeID u, const NodeID max_num_neighbors, Lambda &&l) const {
    __builtin_unreachable();
  }

  //
  // Parallel iteration
  //

  template <typename Lambda> inline void pfor_nodes(Lambda &&l) const {
    __builtin_unreachable();
  }

  template <typename Lambda> inline void pfor_edges(Lambda &&l) const {
    __builtin_unreachable();
  }

  template <typename Lambda>
  inline void pfor_neighbors(
      const NodeID u, const NodeID max_num_neighbors, const NodeID grainsize, Lambda &&l
  ) const {
    __builtin_unreachable();
  }

  //
  // Graph permutation
  //

  inline void set_permutation(StaticArray<NodeID> permutation) final {
    __builtin_unreachable();
  }

  [[nodiscard]] inline bool permuted() const final {
    return false;
  }

  [[nodiscard]] inline NodeID map_original_node(const NodeID u) const final {
    __builtin_unreachable();
  }

  [[nodiscard]] inline StaticArray<NodeID> &&take_raw_permutation() final {
    __builtin_unreachable();
  }

  //
  // Degree buckets
  //

  [[nodiscard]] inline bool sorted() const final {
    return false;
  }

  [[nodiscard]] inline std::size_t number_of_buckets() const final {
    return 1;
  }

  [[nodiscard]] inline std::size_t bucket_size(const std::size_t bucket) const final {
    return _num_nodes;
  }

  [[nodiscard]] inline NodeID first_node_in_bucket(const std::size_t bucket) const final {
    return 0;
  }

  [[nodiscard]] inline NodeID first_invalid_node_in_bucket(const std::size_t bucket) const final {
    return _num_nodes;
  }

  //
  // Access to the underlying graph
  //

  [[nodiscard]] const Graph *underlying_graph() const {
    return _graph;
  }

  [[nodiscard]] StaticArray<BlockID> partition() {
    return StaticArray<BlockID>(_partition.size(), _partition.data());
  }

  [[nodiscard]] const StaticArray<NodeID> &global_to_local() const {
    return _global_to_local;
  }

  [[nodiscard]] StaticArray<NodeID> &global_to_local() {
    return _global_to_local;
  }

  [[nodiscard]] const LocalToGlobalMapping &local_to_global() const {
    return _local_to_global;
  }

  [[nodiscard]] LocalToGlobalMapping &local_to_global() {
    return _local_to_global;
  }

private:
  const Graph *_graph;
  StaticArray<BlockID> _partition;

  const BlockID _block;
  const NodeID _num_nodes;
  const EdgeID _num_edges;
  const NodeWeight _max_node_weight;
  const NodeWeight _total_node_weight;
  const EdgeWeight _total_edge_weight;

  StaticArray<NodeID> _global_to_local;
  LocalToGlobalMapping _local_to_global;
};

} // namespace kaminpar::shm
