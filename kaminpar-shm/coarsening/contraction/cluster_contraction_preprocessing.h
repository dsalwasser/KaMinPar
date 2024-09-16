/*******************************************************************************
 * Common preprocessing utilities for cluster contraction implementations.
 *
 * @file:   cluster_contraction_preprocessing.h
 * @author: Daniel Seemaier
 * @author: Daniel Salwasser
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/coarsening/contraction/cluster_contraction.h"
#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm::contraction {
class CoarseGraphImpl : public CoarseGraph {
public:
  CoarseGraphImpl(Graph graph, StaticArray<NodeID> mapping)
      : _graph(std::move(graph)),
        _mapping(std::move(mapping)) {}

  [[nodiscard]] const Graph &get() const final {
    return _graph;
  }

  Graph &get() final {
    return _graph;
  }

  void project(const StaticArray<BlockID> &array, StaticArray<BlockID> &onto) final {
    tbb::parallel_for<std::size_t>(0, onto.size(), [&](const std::size_t i) {
      onto[i] = array[_mapping[i]];
    });
  }

private:
  Graph _graph;
  StaticArray<NodeID> _mapping;
};

std::tuple<bool, NodeID, StaticArray<NodeID>> contraction_preprocessing(
    const Context &ctx, const Graph &graph, StaticArray<NodeID> clustering, MemoryContext &m_ctx
);
} // namespace kaminpar::shm::contraction
