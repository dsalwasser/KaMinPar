/*******************************************************************************
 * @file:   jet_refiner.h
 * @author: Daniel Seemaier
 * @date:   02.05.2023
 * @brief:  Distributed JET refiner.
 ******************************************************************************/
#pragma once

#include <tbb/concurrent_vector.h>

#include "dkaminpar/context.h"
#include "dkaminpar/datastructures/distributed_graph.h"
#include "dkaminpar/refinement/refiner.h"

#include "common/logger.h"
#include "common/parallel/atomic.h"

namespace kaminpar::dist {
class JetRefiner : public Refiner {
  SET_STATISTICS(true);
  SET_DEBUG(false);

public:
  JetRefiner(const Context &ctx);

  JetRefiner(const JetRefiner &) = delete;
  JetRefiner &operator=(const JetRefiner &) = delete;
  JetRefiner(JetRefiner &&) noexcept = default;
  JetRefiner &operator=(JetRefiner &&) = delete;

  void initialize(const DistributedGraph &graph) final {}

  void refine(
      DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx
  ) final;

private:
  const Context &_ctx;
};
} // namespace kaminpar::dist

