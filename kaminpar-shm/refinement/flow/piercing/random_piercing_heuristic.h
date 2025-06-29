#pragma once

#include <span>

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/flow/piercing/multiway_piercing_heuristic.h"
#include "kaminpar-shm/refinement/flow/util/terminal_sets.h"

#include "kaminpar-common/datastructures/marker.h"
#include "kaminpar-common/datastructures/scalable_vector.h"

namespace kaminpar::shm {

class RandomPiercingHeuristic : public MultiwayPiercingHeuristic {
public:
  RandomPiercingHeuristic(const MultiwayPiercingHeuristicContext &ctx);

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
  const TerminalSets *_terminal_sets;

  ScalableVector<NodeID> _piercing_nodes_candidates;
  Marker<> _piercing_nodes_candidates_marker;

  ScalableVector<BulkPiercingContext> _bulk_piercing_ctx;
};

} // namespace kaminpar::shm
