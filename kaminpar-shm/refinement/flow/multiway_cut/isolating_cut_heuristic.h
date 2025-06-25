#pragma once

#include <memory>
#include <span>
#include <unordered_set>

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/flow/max_flow/max_flow_algorithm.h"
#include "kaminpar-shm/refinement/flow/multiway_cut/multiway_cut_algorithm.h"
#include "kaminpar-shm/refinement/flow/util/terminal_sets.h"

#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm {

class IsolatingCutHeuristic : public MultiwayCutAlgorithm {
  struct Cut {
    EdgeWeight value;
    std::unordered_set<EdgeID> edges;
  };

public:
  using MultiwayCutAlgorithm::compute;

  IsolatingCutHeuristic(const IsolatingCutHeuristicContext &ctx);

  [[nodiscard]] MultiwayCutAlgorithm::Result compute(
      const PartitionedCSRGraph &p_graph,
      const CSRGraph &graph,
      std::span<const NodeID> reverse_edges,
      const TerminalSets &terminal_sets
  ) override;

private:
  Cut compute_cut(
      const BlockID block,
      const std::unordered_set<NodeID> &terminals,
      std::span<const EdgeWeight> flow
  );

  EdgeWeight compute_cut_value(const std::unordered_set<EdgeID> &cut_edges);

private:
  const IsolatingCutHeuristicContext &_ctx;

  const CSRGraph *_graph;
  std::span<const NodeID> _reverse_edges;

  std::unique_ptr<MaxFlowAlgorithm> _max_flow_algorithm;
  StaticArray<BlockID> _node_assignment;
};

} // namespace kaminpar::shm