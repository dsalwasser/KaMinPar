/*******************************************************************************
 * Initial partitioner that invokes KaMinPar.
 *
 * @file:   kaminpar_initial_partitioner.cc
 * @author: Daniel Seemaier
 * @date:   06.11.2021
 ******************************************************************************/
#include "kaminpar-dist/initial_partitioning/kaminpar_initial_partitioner.h"

#include "kaminpar-dist/logger.h"

#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/factories.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/scalable_vector.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::dist {

namespace {

SET_DEBUG(false);

}

shm::PartitionedGraph KaMinParInitialPartitioner::initial_partition(
    const shm::Graph &graph, const shm::PartitionContext &p_ctx
) {
  if (graph.n() <= 1 || p_ctx.k == 1) {
    return {graph, p_ctx.k, StaticArray<BlockID>(graph.n())};
  }

  auto shm_ctx = _ctx.initial_partitioning.kaminpar;
  shm_ctx.refinement.lp.num_iterations = 1;
  shm_ctx.partition = p_ctx;
  shm_ctx.compression.setup(graph);

  /*
  std::vector<BlockWeight> max_block_weights;
  for (BlockID b = 0; b < shm_ctx.partition.k; ++b) {
    max_block_weights.push_back(shm_ctx.partition.max_block_weight(b));
  }
  DBG << "Max block weights for initial partitioning: " << max_block_weights
      << ", total node weight of our subgraph: " << graph.total_node_weight();
  */

  DISABLE_TIMERS();
  START_HEAP_PROFILER("KaMinPar");

  const bool was_quiet = Logger::is_quiet();
  Logger::set_quiet_mode(true);
  auto p_graph = shm::factory::create_partitioner(graph, shm_ctx)->partition();
  Logger::set_quiet_mode(was_quiet);

  STOP_HEAP_PROFILER();
  ENABLE_TIMERS();

  return p_graph;
}

} // namespace kaminpar::dist
