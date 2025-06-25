#pragma once

#include <algorithm>
#include <span>

#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/dynamic_map.h"
#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm {

class TerminalSets {
public:
  void initialize(const BlockID num_terminal_sets, const NodeID num_nodes) {
    _num_terminal_sets = num_terminal_sets;
    _num_nodes = num_nodes;

    _num_terminals = 0;

    if (_node_status.size() < num_nodes) {
      _node_status.resize(num_nodes, static_array::noinit);
    }
    std::fill_n(_node_status.begin(), num_nodes, kInvalidBlockID);

    _terminal_sets.clear();
    _terminal_sets.resize(num_terminal_sets);
  }

  [[nodiscard]] BlockID num_terminal_sets() const {
    return _num_terminal_sets;
  }

  [[nodiscard]] NodeID num_terminals() const {
    return _num_terminals;
  }

  [[nodiscard]] bool is_terminal(const NodeID u) const {
    KASSERT(u < _num_nodes);
    return _node_status[u] != kInvalidBlockID;
  }

  [[nodiscard]] BlockID terminal_set(const NodeID u) const {
    KASSERT(u < _num_nodes);
    return _node_status[u];
  }

  [[nodiscard]] std::span<const NodeID> terminal_set_nodes(const BlockID terminal_set) const {
    KASSERT(terminal_set < _num_terminal_sets);
    return _terminal_sets[terminal_set];
  }

  void set_terminal_set(const NodeID u, const BlockID terminal_set) {
    KASSERT(terminal_set < _num_terminal_sets);
    KASSERT(u < _num_nodes);
    KASSERT(!is_terminal(u));

    _num_terminals += 1;
    _node_status[u] = terminal_set;
    _terminal_sets[terminal_set].push_back(u);
  }

private:
  BlockID _num_terminal_sets;
  NodeID _num_nodes;

  NodeID _num_terminals;
  StaticArray<BlockID> _node_status;
  ScalableVector<ScalableVector<NodeID>> _terminal_sets;
};

} // namespace kaminpar::shm
