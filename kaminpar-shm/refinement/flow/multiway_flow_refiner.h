/*******************************************************************************
 * Multi-way flow refiner.
 *
 * @file:   multiway_flow_refiner.h
 * @author: Daniel Salwasser
 * @date:   10.04.2025
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/refiner.h"

namespace kaminpar::shm {

class MultiwayFlowRefiner : public Refiner {
  SET_DEBUG(true);

  struct FlowNetwork {
    CSRGraph graph;
    StaticArray<NodeID> reverse_edges;
    std::unordered_map<NodeID, NodeID> global_to_local_mapping;
  };

  struct Cut {
    NodeWeight weight;
    std::unordered_set<NodeID> nodes;
  };

public:
  MultiwayFlowRefiner(const Context &ctx);
  ~MultiwayFlowRefiner() override;

  MultiwayFlowRefiner(const MultiwayFlowRefiner &) = delete;
  MultiwayFlowRefiner &operator=(const MultiwayFlowRefiner &) = delete;

  MultiwayFlowRefiner(MultiwayFlowRefiner &&) noexcept = default;
  MultiwayFlowRefiner &operator=(MultiwayFlowRefiner &&) noexcept = default;

  [[nodiscard]] std::string name() const override;

  void initialize(const PartitionedGraph &p_graph) override;

  bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) override;

private:
  bool refine(PartitionedCSRGraph &p_graph, const CSRGraph &graph, const PartitionContext &p_ctx);

private:
  const MultiwayFlowRefinementContext &_f_ctx;
};

} // namespace kaminpar::shm
