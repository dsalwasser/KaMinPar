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

#include "kaminpar-common/datastructures/fast_reset_array.h"
#include "kaminpar-common/datastructures/scalable_vector.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/logger.h"

namespace kaminpar::shm {

class LabellingFunctionHeuristic : public MultiwayCutAlgorithm {
  SET_DEBUG(false);

  using Neighborhood = ScalableVector<std::pair<NodeID, EdgeWeight>>;
  using EdgeCollector = FastResetArray<EdgeWeight, NodeID>;

  struct FlowNetwork {
    NodeID source;
    NodeID sink;

    CSRGraph graph;
    StaticArray<NodeID> reverse_edges;

    NodeID node_start;
    NodeID node_end;
    StaticArray<NodeID> remapping;
  };

public:
  LabellingFunctionHeuristic(const LabellingFunctionHeuristicContext &ctx);

  [[nodiscard]] MultiwayCutAlgorithm::Result compute(
      const PartitionedCSRGraph &p_graph,
      const CSRGraph &graph,
      std::span<const NodeID> reverse_edges,
      const TerminalSets &terminal_sets
  ) override;

private:
  void initialize_labelling_function();

  void improve_labelling_function();

  EdgeWeight compute_labelling_function_cost() const;

  FlowNetwork construct_flow_network(BlockID label);

  void derive_labelling_function(
      BlockID label, const FlowNetwork &flow_network, std::span<const EdgeWeight> flow
  );

  std::unordered_set<EdgeID> derive_cut_edges() const;

  std::unordered_set<NodeID> static compute_cut_nodes(
      const CSRGraph &graph, const NodeID terminal, std::span<const EdgeWeight> flow
  );

  bool is_valid_labelling_function() const;

private:
  const LabellingFunctionHeuristicContext &_ctx;

  const PartitionedCSRGraph *_p_graph;
  const CSRGraph *_graph;
  const TerminalSets *_terminal_sets;

  EdgeWeight _labelling_function_cost;
  StaticArray<BlockID> _labelling_function;

  ScalableVector<Neighborhood> _terminal_tneighborhoods;
  ScalableVector<Neighborhood> _terminal_neighborhoods;
  EdgeCollector _edge_collector;

  std::unique_ptr<MaxFlowAlgorithm> _max_flow_algorithm;
};

} // namespace kaminpar::shm