/*******************************************************************************
 * Initial partitioner that uses Mt-KaHypar. Only available if the Mt-KaHyPar
 * library is installed on the system.
 *
 * @file:   mtkahypar_initial_partitioner.cc
 * @author: Daniel Seemaier
 * @date:   15.09.2022
 ******************************************************************************/
#include "kaminpar-dist/initial_partitioning/mtkahypar_initial_partitioner.h"

#ifdef KAMINPAR_HAVE_MTKAHYPAR_LIB
#include <cstdio>
#include <filesystem>
#include <fstream>

#include <libmtkahypar.h>

#include "kaminpar-common/datastructures/noinit_vector.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/parallel/algorithm.h"
#include "kaminpar-common/random.h"
#include "kaminpar-common/timer.h"
#endif // KAMINPAR_HAVE_MTKAHYPAR_LIB

#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/assert.h"

namespace kaminpar::dist {

shm::PartitionedGraph MtKaHyParInitialPartitioner::initial_partition(
    [[maybe_unused]] const shm::Graph &graph, [[maybe_unused]] const shm::PartitionContext &p_ctx
) {
#ifdef KAMINPAR_HAVE_MTKAHYPAR_LIB
  mt_kahypar_context_t *mt_kahypar_ctx = mt_kahypar_context_new();
  mt_kahypar_load_preset(mt_kahypar_ctx, DEFAULT);
  mt_kahypar_set_partitioning_parameters(
      mt_kahypar_ctx, static_cast<mt_kahypar_partition_id_t>(p_ctx.k), p_ctx.epsilon(), KM1
  );
  mt_kahypar_set_seed(Random::get_seed());
  mt_kahypar_set_context_parameter(mt_kahypar_ctx, VERBOSE, "0");

  mt_kahypar_initialize_thread_pool(_ctx.parallel.num_threads, true);

  const mt_kahypar_hypernode_id_t num_vertices = graph.n();
  const mt_kahypar_hyperedge_id_t num_edges = graph.m() / 2; // Only need one direction

  NoinitVector<EdgeID> edge_position(2 * num_edges);
  graph.reified([&](const auto &graph) {
    graph.pfor_nodes([&](const NodeID u) {
      graph.neighbors(u, [&](const EdgeID e, const NodeID v) { edge_position[e] = u < v; });
    });
  });
  parallel::prefix_sum(edge_position.begin(), edge_position.end(), edge_position.begin());

  NoinitVector<mt_kahypar_hypernode_id_t> edges(2 * num_edges);
  NoinitVector<mt_kahypar_hypernode_weight_t> edge_weights(num_edges);
  NoinitVector<mt_kahypar_hypernode_weight_t> vertex_weights(num_vertices);
  edges.reserve(2 * num_edges);
  edge_weights.reserve(num_edges);

  graph.reified([&](const auto &graph) {
    graph.pfor_nodes([&](const NodeID u) {
      vertex_weights[u] = static_cast<mt_kahypar_hypernode_weight_t>(graph.node_weight(u));

      graph.neighbors(u, [&](const EdgeID e, const NodeID v, const EdgeWeight w) {
        if (v < u) { // Only need edges in one direction
          return;
        }

        EdgeID position = edge_position[e] - 1;
        edges[2 * position] = static_cast<mt_kahypar_hypernode_id_t>(u);
        edges[2 * position + 1] = static_cast<mt_kahypar_hypernode_id_t>(v);
        edge_weights[position] = static_cast<mt_kahypar_hypernode_weight_t>(w);
      });
    });
  });

  mt_kahypar_hypergraph_t mt_kahypar_graph = mt_kahypar_create_graph(
      DEFAULT, num_vertices, num_edges, edges.data(), edge_weights.data(), vertex_weights.data()
  );

  mt_kahypar_partitioned_hypergraph_t mt_kahypar_partitioned_graph =
      mt_kahypar_partition(mt_kahypar_graph, mt_kahypar_ctx);

  NoinitVector<mt_kahypar_partition_id_t> partition(num_vertices);
  mt_kahypar_get_partition(mt_kahypar_partitioned_graph, partition.data());

  // Copy partition to BlockID vector
  StaticArray<BlockID> partition_cpy(num_vertices);
  tbb::parallel_for<std::size_t>(0, num_vertices, [&](const std::size_t i) {
    partition_cpy[i] = static_cast<BlockID>(partition[i]);
  });

  mt_kahypar_free_partitioned_hypergraph(mt_kahypar_partitioned_graph);
  mt_kahypar_free_hypergraph(mt_kahypar_graph);
  mt_kahypar_free_context(mt_kahypar_ctx);

  return {graph, p_ctx.k, std::move(partition_cpy)};
#else  // KAMINPAR_HAVE_MTKAHYPAR_LIB
  KASSERT(false, "Mt-KaHyPar initial partitioner is not available.", assert::always);
  __builtin_unreachable();
#endif // KAMINPAR_HAVE_MTKAHYPAR_LIB
}

} // namespace kaminpar::dist
