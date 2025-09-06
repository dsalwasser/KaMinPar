#pragma once

#ifdef KAMINPAR_WHFC_FOUND
#include "algorithm/hyperflowcutter.h"
#include "algorithm/parallel_push_relabel.h"
#include "algorithm/sequential_push_relabel.h"
#include "assert.h"
#include "datastructure/flow_hypergraph_builder.h"
#include "definitions.h"

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/datastructures/delta_partitioned_graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/flow/flow_cutter/flow_cutter_algorithm.h"
#include "kaminpar-shm/refinement/flow/flow_network/border_region.h"
#include "kaminpar-shm/refinement/flow/flow_network/flow_network.h"
#include "kaminpar-shm/refinement/flow/rebalancer/flow_rebalancer.h"
#include "kaminpar-shm/refinement/flow/util/breadth_first_search.h"
#include "kaminpar-shm/refinement/gains/delta_gain_caches.h"
#include "kaminpar-shm/refinement/gains/sparse_gain_cache.h"

#include "kaminpar-common/datastructures/marker.h"
#include "kaminpar-common/datastructures/scalable_vector.h"
#include "kaminpar-common/logger.h"

namespace kaminpar::shm {

class HyperFlowCutter : public FlowCutterAlgorithm {
  SET_DEBUG(false);

  static constexpr bool kSourceTag = true;
  static constexpr bool kSinkTag = false;

  using GainCache = NormalSparseGainCache<CSRGraph, PartitionedCSRGraph, DeltaPartitionedCSRGraph>;
  using DeltaGainCache = GenericDeltaGainCache<GainCache>;

  using RebalanceResult = FlowRebalancer::Result;

public:
  HyperFlowCutter(
      const PartitionContext &p_ctx,
      const FlowCutterContext &fc_ctx,
      bool run_sequentially,
      const PartitionedCSRGraph &p_graph,
      GainCache &gain_cache
  );

  [[nodiscard]] virtual Result
  compute_cut(const BorderRegion &border_region, const FlowNetwork &flow_network) override;

  void free() override;

private:
  void construct_hypergraph(const FlowNetwork &flow_network);

  Result run_hyper_flow_cutter(const BorderRegion &border_region, const FlowNetwork &flow_network);

  void compute_distances(
      const BorderRegion &border_region,
      const FlowNetwork &flow_network,
      std::vector<whfc::HopDistance> &distances
  );

  template <typename CutterState>
  void compute_moves(
      const BorderRegion &border_region,
      const FlowNetwork &flow_network,
      const CutterState &cutter_state
  );

  template <typename CutterState>
  void rebalance(
      bool source_side_cut,
      bool rebalance_source_side,
      EdgeWeight cur_cut_value,
      const CutterState &cutter_state,
      const BorderRegion &border_region,
      const FlowNetwork &flow_network
  );

private:
  const PartitionContext &_p_ctx;
  const FlowCutterContext &_fc_ctx;
  const bool _run_sequentially;

  whfc::FlowHypergraphBuilder _hypergraph;
  whfc::HyperFlowCutter<whfc::SequentialPushRelabel> _sequential_flow_cutter;
  whfc::HyperFlowCutter<whfc::ParallelPushRelabel> _parallel_flow_cutter;

  Marker<> _bfs_marker;
  BFSRunner _bfs_runner;

  DeltaPartitionedCSRGraph _delta_p_graph;
  DeltaGainCache _delta_gain_cache;
  std::unique_ptr<FlowRebalancer> _flow_rebalancer;

  EdgeWeight _gain;
  bool _improve_balance;
  bool _time_limit_exceeded;
  ScalableVector<Move> _moves;
};

} // namespace kaminpar::shm

#endif
