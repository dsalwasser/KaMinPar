#pragma once

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/flow/piercing/multiway_piercing_heuristic.h"
#include "kaminpar-shm/refinement/flow/util/terminal_sets.h"

#include "kaminpar-common/datastructures/binary_heap.h"
#include "kaminpar-common/datastructures/marker.h"
#include "kaminpar-common/datastructures/scalable_vector.h"

namespace kaminpar::shm {

class RelativeGainPiercingHeuristic : public MultiwayPiercingHeuristic {
  using RelativeGain = float;
  using PriorityQueue = BinaryMaxHeap<RelativeGain>;

public:
  RelativeGainPiercingHeuristic(const MultiwayPiercingHeuristicContext &ctx);

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
  const MultiwayPiercingHeuristicContext &_ctx;

  const CSRGraph *_graph;
  const PartitionedCSRGraph *_p_graph;
  const TerminalSets *_terminal_sets;

  ScalableVector<NodeID> _piercing_nodes;

  PriorityQueue _priority_queue;

  ScalableVector<BulkPiercingContext> _bulk_piercing_ctx;
};

} // namespace kaminpar::shm
