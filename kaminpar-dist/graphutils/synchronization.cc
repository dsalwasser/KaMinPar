/*******************************************************************************
 * Implements common synchronization operations for distributed graphs.
 *
 * @file:   graph_synchronization.cc
 * @author: Daniel Seemaier
 * @date:   15.07.2022
 ******************************************************************************/
#include "kaminpar-dist/graphutils/synchronization.h"

#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"
#include "kaminpar-dist/graphutils/communication.h"

namespace kaminpar::dist::graph {

void synchronize_ghost_node_block_ids(DistributedPartitionedGraph &p_graph) {
  struct Message {
    NodeID node;
    BlockID block;
  };

  mpi::graph::sparse_alltoall_interface_to_pe<Message>(
      p_graph.graph(),
      [&](const NodeID u) -> Message { return {.node = u, .block = p_graph.block(u)}; },
      [&](const auto &recv_buffer, const PEID pe) {
        tbb::parallel_for<std::size_t>(0, recv_buffer.size(), [&](const std::size_t i) {
          const auto [local_node_on_pe, block] = recv_buffer[i];
          const auto global_node =
              static_cast<GlobalNodeID>(p_graph.offset_n(pe) + local_node_on_pe);
          const NodeID local_node = p_graph.global_to_local_node(global_node);
          p_graph.set_block<false>(local_node, block);
        });
      }
  );
}

} // namespace kaminpar::dist::graph
