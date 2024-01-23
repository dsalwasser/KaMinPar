/******************************************************************************
 * Label propagation for graph coarsening / clustering.
 *
 * @file:   lp_clustering.cc
 * @author: Daniel Seemaier
 * @date:   29.09.2021
 ******************************************************************************/
#include "kaminpar-shm/coarsening/lp_clustering.h"

namespace kaminpar::shm {

LPClustering::LPClustering(const NodeID max_n, const CoarseningContext &c_ctx)
    : _csr_core{std::make_unique<LPClusteringImpl<CSRGraph>>(max_n, c_ctx)},
      _compact_csr_core{std::make_unique<LPClusteringImpl<CompactCSRGraph>>(max_n, c_ctx)},
      _compressed_core{std::make_unique<LPClusteringImpl<CompressedGraph>>(max_n, c_ctx)} {}

// We must declare the destructor explicitly here, otherwise, it is implicitly generated before
// LabelPropagationClusterCore is complete.
LPClustering::~LPClustering() = default;

void LPClustering::set_max_cluster_weight(const NodeWeight max_cluster_weight) {
  _csr_core->set_max_cluster_weight(max_cluster_weight);
  _compact_csr_core->set_max_cluster_weight(max_cluster_weight);
  _compressed_core->set_max_cluster_weight(max_cluster_weight);
}

void LPClustering::set_desired_cluster_count(const NodeID count) {
  _csr_core->set_desired_num_clusters(count);
  _compact_csr_core->set_desired_num_clusters(count);
  _compressed_core->set_desired_num_clusters(count);
}

const Clusterer::AtomicClusterArray &LPClustering::compute_clustering(const Graph &graph) {
  // Compute a clustering and setup/release the data structures used by the core, so that they can
  // be shared by all graph implementations.
  const auto compute = [&](auto &core, auto &graph) {
    if (!_allocated) {
      _allocated = true;
      core.allocate();
    } else {
      core.setup(std::move(_structs));
      core.setup_clusters(std::move(_clusters));
      core.setup_cluster_weights(std::move(_cluster_weights));
    }

    core.compute_clustering(graph);

    _structs = core.release();
    _clusters = core.take_clusters();
    _cluster_weights = core.take_cluster_weights();
  };

  if (auto *csr_graph = dynamic_cast<CSRGraph *>(graph.underlying_graph()); csr_graph != nullptr) {
    compute(*_csr_core, *csr_graph);
  } else if (auto *compact_csr_graph = dynamic_cast<CompactCSRGraph *>(graph.underlying_graph());
             compact_csr_graph != nullptr) {
    compute(*_compact_csr_core, *compact_csr_graph);
  } else if (auto *compressed_graph = dynamic_cast<CompressedGraph *>(graph.underlying_graph());
             compressed_graph != nullptr) {
    compute(*_compressed_core, *compressed_graph);
  }

  return _clusters;
}

} // namespace kaminpar::shm
