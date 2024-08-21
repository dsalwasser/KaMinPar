/*******************************************************************************
 * Initial refiner that does nothing.
 *
 * @file:   initial_noop_refiner.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#include "kaminpar-shm/initial_partitioning/initial_noop_refiner.h"

namespace kaminpar::shm {
void InitialNoopRefiner::init(const Graph &) {}

bool InitialNoopRefiner::refine(PartitionedGraph &, const PartitionContext &) {
  return false;
}
} // namespace kaminpar::shm
