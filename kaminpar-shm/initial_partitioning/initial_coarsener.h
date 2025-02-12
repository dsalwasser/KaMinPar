/*******************************************************************************
 * Sequential label propagation coarsening used during initial bipartitionign.
 *
 * @file:   initial_coarsener.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include <utility>

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/initial_partitioning/sequential_graph_hierarchy.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/fast_reset_array.h"
#include "kaminpar-common/datastructures/scalable_vector.h"
#include "kaminpar-common/random.h"

namespace kaminpar::shm {

struct InitialCoarsenerTimings {
  std::uint64_t contract_ms = 0;
  std::uint64_t alloc_ms = 0;
  std::uint64_t interleaved1_ms = 0;
  std::uint64_t interleaved2_ms = 0;
  std::uint64_t lp_ms = 0;
  std::uint64_t total_ms = 0;

  InitialCoarsenerTimings &operator+=(const InitialCoarsenerTimings &other) {
    contract_ms += other.contract_ms;
    alloc_ms += other.alloc_ms;
    interleaved1_ms += other.interleaved1_ms;
    interleaved2_ms += other.interleaved2_ms;
    lp_ms += other.lp_ms;
    total_ms += other.total_ms;
    return *this;
  }
};

class InitialCoarsener {
  static constexpr std::size_t kChunkSize = 256;
  static constexpr std::size_t kNumberOfNodePermutations = 16;

  using ContractionResult = std::pair<CSRGraph, StaticArray<NodeID>>;

public:
  struct Cluster {
    // Steal one bit from the weight field to achieve 8 instead 12 bytes per entry
    bool locked : 1;
    NodeWeight weight : std::numeric_limits<NodeWeight>::digits - 1;
    NodeID leader;
  };

  static_assert(
      sizeof(NodeWeight) != sizeof(NodeID) || sizeof(Cluster) == sizeof(NodeWeight) + sizeof(NodeID)
  );

  InitialCoarsener(const InitialCoarseningContext &c_ctx);

  InitialCoarsener(const InitialCoarsener &) = delete;
  InitialCoarsener &operator=(const InitialCoarsener &) = delete;

  InitialCoarsener(InitialCoarsener &&) noexcept = delete;
  InitialCoarsener &operator=(InitialCoarsener &&) = delete;

  [[nodiscard]] inline std::size_t level() const {
    return _hierarchy.level();
  }

  [[nodiscard]] inline bool empty() const {
    return _hierarchy.empty();
  }

  [[nodiscard]] inline const CSRGraph *current() const {
    return &_hierarchy.current();
  }

  void init(const CSRGraph &graph);

  const CSRGraph *coarsen(NodeWeight max_cluster_weight);

  PartitionedCSRGraph uncoarsen(PartitionedCSRGraph &&c_p_graph);

  void reset_current_clustering();

  template <typename Weights>
  void reset_current_clustering(const NodeID n, const Weights &node_weights) {
    KASSERT(n <= _clustering.size());
    KASSERT(n <= node_weights.size());

    _current_num_moves = 0;
    for (NodeID u = 0; u < n; ++u) {
      _clustering[u].locked = false;
      _clustering[u].leader = u;
      _clustering[u].weight = node_weights[u];
    }
  }

  void reset_current_clustering_unweighted(const NodeID n, const NodeWeight unit_node_weight);

  void handle_node(NodeID u, NodeWeight max_cluster_weight);
  NodeID pick_cluster(NodeID u, NodeWeight u_weight, NodeWeight max_cluster_weight);
  NodeID pick_cluster_from_rating_map(NodeID u, NodeWeight u_weight, NodeWeight max_cluster_weight);

  InitialCoarsenerTimings timings() {
    auto timings = _timings;
    _timings = InitialCoarsenerTimings{};
    return timings;
  }

private:
  ContractionResult contract_current_clustering();

  void perform_label_propagation(NodeWeight max_cluster_weight);

  void interleaved_handle_node(NodeID c_u, NodeWeight c_u_weight);

  void interleaved_visit_neighbor(NodeID, NodeID c_v, EdgeWeight weight);

  const CSRGraph *_input_graph;
  const CSRGraph *_current_graph;
  SequentialGraphHierarchy _hierarchy;

  const InitialCoarseningContext &_c_ctx;

  ScalableVector<Cluster> _clustering;
  FastResetArray<EdgeWeight> _rating_map;
  ScalableVector<NodeID> _cluster_sizes;
  ScalableVector<NodeID> _leader_node_mapping;
  FastResetArray<EdgeWeight> _edge_weight_collector;
  ScalableVector<NodeID> _cluster_nodes;

  NodeID _current_num_moves = 0;
  bool _precomputed_clustering = false;
  NodeWeight _interleaved_max_cluster_weight = 0;
  bool _interleaved_locked = false;

  Random &_rand = Random::instance();
  RandomPermutations<NodeID, kChunkSize, kNumberOfNodePermutations> _random_permutations{_rand};
  std::vector<NodeID> _chunks;

  InitialCoarsenerTimings _timings{};
};

} // namespace kaminpar::shm
