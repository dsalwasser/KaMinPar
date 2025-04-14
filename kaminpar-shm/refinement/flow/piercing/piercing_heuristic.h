#pragma once

#include <unordered_set>

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/logger.h"

namespace kaminpar::shm {

class PiercingHeuristic {
  SET_DEBUG(false);

public:
  PiercingHeuristic(
      const CSRGraph &graph,
      const std::unordered_set<NodeID> &initial_source_side_nodes,
      const std::unordered_set<NodeID> &initial_sink_side_nodes
  );

  NodeID pierce_on_source_side(
      const std::unordered_set<NodeID> &source_side_cut,
      const std::unordered_set<NodeID> &sink_side_cut,
      NodeWeight max_piercing_node_weight
  );

  NodeID pierce_on_sink_side(
      const std::unordered_set<NodeID> &source_side_cut,
      const std::unordered_set<NodeID> &sink_side_cut,
      NodeWeight max_piercing_node_weight
  );

private:
  NodeID find_piercing_node(
      const std::unordered_set<NodeID> &terminal_cut,
      const std::unordered_set<NodeID> &other_terminal_cut,
      const std::unordered_set<NodeID> &initial_terminal_side_nodes,
      const NodeWeight max_piercing_node_weight
  );

  void compute_distances();

  [[nodiscard]] StaticArray<NodeID> compute_distances(NodeID terminal);

private:
  const CSRGraph &_graph;
  const std::unordered_set<NodeID> &_initial_source_side_nodes;
  const std::unordered_set<NodeID> &_initial_sink_side_nodes;

  StaticArray<NodeID> _distance;
};

} // namespace kaminpar::shm
