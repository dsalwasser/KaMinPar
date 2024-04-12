/*******************************************************************************
 * Coarsener that is optimized to contract clusterings.
 *
 * @file:   cluster_coarsener.h
 * @author: Daniel Seemaier
 * @date:   29.09.2021
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/coarsening/clusterer.h"
#include "kaminpar-shm/coarsening/coarsener.h"
#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/coarsening/contraction/cluster_contraction.h"
#include "kaminpar-shm/kaminpar.h"

namespace kaminpar::shm {
class ClusteringCoarsener : public Coarsener {
public:
  ClusteringCoarsener(
      std::unique_ptr<Clusterer> clustering_algorithm,
      const Graph &input_graph,
      const CoarseningContext &c_ctx
  )
      : _input_graph(input_graph),
        _current_graph(&input_graph),
        _clustering_algorithm(std::move(clustering_algorithm)),
        _c_ctx(c_ctx) {}

  ClusteringCoarsener(const ClusteringCoarsener &) = delete;
  ClusteringCoarsener &operator=(const ClusteringCoarsener) = delete;

  ClusteringCoarsener(ClusteringCoarsener &&) = delete;
  ClusteringCoarsener &operator=(ClusteringCoarsener &&) = delete;

  std::pair<const Graph *, bool> compute_coarse_graph(
      NodeWeight max_cluster_weight, NodeID to_size, const bool free_memory_afterwards
  ) final;
  PartitionedGraph uncoarsen(PartitionedGraph &&p_graph) final;

  [[nodiscard]] const Graph *coarsest_graph() const final {
    return _current_graph;
  }

  [[nodiscard]] std::size_t size() const final {
    return _hierarchy.size();
  }

  void initialize(const Graph *) final {}

  [[nodiscard]] const CoarseningContext &context() const {
    return _c_ctx;
  }

private:
  const Graph &_input_graph;
  const Graph *_current_graph;

  std::vector<std::unique_ptr<CoarseGraph>> _hierarchy;

  std::unique_ptr<Clusterer> _clustering_algorithm;

  const CoarseningContext &_c_ctx;
  contraction::MemoryContext _contraction_m_ctx{};
};
} // namespace kaminpar::shm
