#pragma once

#include <span>
#include <unordered_set>
#include <vector>

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/flow/util/terminal_sets.h"

namespace kaminpar::shm {

class MultiwayCutAlgorithm {
public:
  struct Result {
    EdgeWeight cut_value;
    std::unordered_set<EdgeID> cut_edges;
  };

  virtual ~MultiwayCutAlgorithm() = default;

  [[nodiscard]] virtual Result compute(
      const PartitionedCSRGraph &p_graph,
      const CSRGraph &graph,
      std::span<const NodeID> reverse_edges,
      const TerminalSets &terminal_sets
  ) = 0;
};

namespace debug {

[[nodiscard]] bool is_valid_multiway_cut(
    const CSRGraph &graph,
    const std::vector<std::unordered_set<NodeID>> &terminal_sets,
    const std::unordered_set<EdgeID> &cut_edges
);

[[nodiscard]] bool is_valid_multiway_cut(
    const CSRGraph &graph,
    const TerminalSets &terminal_sets,
    const std::unordered_set<EdgeID> &cut_edges
);

} // namespace debug

} // namespace kaminpar::shm