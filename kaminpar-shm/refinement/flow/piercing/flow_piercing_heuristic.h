#pragma once

#include <span>

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/flow/max_flow/fifo_preflow_push_algorithm.h"
#include "kaminpar-shm/refinement/flow/piercing/multiway_piercing_heuristic.h"
#include "kaminpar-shm/refinement/flow/util/node_status.h"
#include "kaminpar-shm/refinement/flow/util/terminal_sets.h"

#include "kaminpar-common/datastructures/marker.h"
#include "kaminpar-common/datastructures/scalable_vector.h"
#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm {

class FlowPiercingHeuristic : public MultiwayPiercingHeuristic {
public:
  FlowPiercingHeuristic(const MultiwayPiercingHeuristicContext &ctx);

  void initialize(
      const CSRGraph &graph,
      const PartitionedCSRGraph &p_graph,
      const PartitionContext &p_ctx,
      const TerminalSets &terminal_sets
  ) override;

  std::span<const NodeID> find_piercing_nodes(
      BlockID terminal_set_to_pierce, const NodeWeight terminal_set_weight, NodeWeight max_weight
  ) override;

private:
  void expand_cut(
      bool source_side, std::span<const EdgeWeight> flow, std::span<const NodeID> border_nodes
  );

  void compute_piercing_node_candidates();

  void compute_piercing_nodes(
      BlockID terminal_set, NodeWeight terminal_set_weight, NodeWeight max_weight
  );

private:
  const MultiwayPiercingHeuristicContext &_ctx;

  const CSRGraph *_graph;
  const TerminalSets *_terminal_sets;

  StaticArray<NodeID> _reverse_edges;

  ScalableVector<NodeID> _piercing_nodes;

  ScalableVector<NodeID> _terminal_set_nodes;
  ScalableVector<NodeID> _other_terminal_set_nodes;
  NodeStatus _node_status;

  ScalableVector<NodeID> _unreachable_piercing_node_candidates;
  ScalableVector<NodeID> _reachable_piercing_node_candidates;
  Marker<> _piercing_nodes_candidates_marker;

  FIFOPreflowPushContext _max_flow_algorithm_ctx;
  FIFOPreflowPushAlgorithm _max_flow_algorithm;

  ScalableVector<BulkPiercingContext> _bulk_piercing_ctx;
};

} // namespace kaminpar::shm
