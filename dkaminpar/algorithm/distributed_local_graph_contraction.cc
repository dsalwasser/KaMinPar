/*******************************************************************************
 * This file is part of KaMinPar.
 *
 * Copyright (C) 2021 Daniel Seemaier <daniel.seemaier@kit.edu>
 *
 * KaMinPar is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * KaMinPar is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with KaMinPar.  If not, see <http://www.gnu.org/licenses/>.
 *
******************************************************************************/
#include "dkaminpar/algorithm/distributed_local_graph_contraction.h"

#include "dkaminpar/utility/mpi_helper.h"
#include "kaminpar/datastructure/rating_map.h"

#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>

namespace dkaminpar::graph {
SET_DEBUG(true);

ContractionResult contract_locally(const DistributedGraph &graph, const scalable_vector<DNodeID> &clustering,
                                   ContractionMemoryContext m_ctx) {
  const auto [size, rank] = mpi::get_comm_info();

  auto &buckets_index = m_ctx.buckets_index;
  auto &buckets = m_ctx.buckets;
  auto &leader_mapping = m_ctx.leader_mapping;
  auto &all_buffered_nodes = m_ctx.all_buffered_nodes;

  scalable_vector<DNodeID> mapping(graph.total_n());
  if (leader_mapping.size() < graph.n()) { leader_mapping.resize(graph.n()); }
  if (buckets.size() < graph.n()) { buckets.resize(graph.n()); }

  //
  // Compute a mapping from the nodes of the current graph to the nodes of the coarse graph
  // I.e., node_mapping[node u] = coarse node c_u
  //

  // Set node_mapping[x] = 1 iff. there is a cluster with leader x
  graph.pfor_nodes([&](const DNodeID u) { leader_mapping[u] = 0; });
  graph.pfor_nodes([&](const DNodeID u) { leader_mapping[clustering[u]].store(1, std::memory_order_relaxed); });

  // Compute prefix sum to get coarse node IDs (starting at 1!)
  shm::parallel::prefix_sum(leader_mapping.begin(), leader_mapping.begin() + graph.n(), leader_mapping.begin());
  const DNodeID c_n = leader_mapping[graph.n() - 1]; // number of nodes in the coarse graph

  // Compute new node distribution, total number of coarse nodes
  DNodeID last_node;
  MPI_Scan(&c_n, &last_node, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
  const DNodeID first_node = last_node - c_n;

  scalable_vector<DNodeID> c_node_distribution(size + 1);
  c_node_distribution[rank + 1] = last_node;
  MPI_Allgather(&c_node_distribution[rank + 1], 1, MPI_UINT64_T, c_node_distribution.data() + 1, 1, MPI_UINT64_T,
                MPI_COMM_WORLD);
  const DNodeID c_global_n = c_node_distribution.back();

  // Assign coarse node ID to all nodes
  graph.pfor_nodes([&](const DNodeID u) { mapping[u] = leader_mapping[clustering[u]]; });
  graph.pfor_nodes([&](const DNodeID u) { --mapping[u]; });

  buckets_index.clear();
  buckets_index.resize(c_n + 1);

  //
  // Sort nodes into buckets: place all nodes belonging to coarse node i into the i-th bucket
  //
  // Count the number of nodes in each bucket, then compute the position of the bucket in the global buckets array
  // using a prefix sum, roughly 2/5-th of time on europe.osm with 2/3-th to 1/3-tel for loop to prefix sum
  graph.pfor_nodes([&](const DNodeID u) { buckets_index[mapping[u]].fetch_add(1, std::memory_order_relaxed); });

  shm::parallel::prefix_sum(buckets_index.begin(), buckets_index.end(), buckets_index.begin());
  ASSERT(buckets_index.back() <= graph.n());

  // Sort nodes into   buckets, roughly 3/5-th of time on europe.osm
  tbb::parallel_for(static_cast<DNodeID>(0), graph.n(), [&](const DNodeID u) {
    const std::size_t pos = buckets_index[mapping[u]].fetch_sub(1, std::memory_order_relaxed) - 1;
    buckets[pos] = u;
  });

  //
  // Build nodes array of the coarse graph
  // - firstly, we count the degree of each coarse node
  // - secondly, we obtain the nodes array using a prefix sum
  //
  scalable_vector<DEdgeID> c_nodes(c_n + 1);
  scalable_vector<DNodeWeight> c_node_weights(c_n); // overallocate

  // Build coarse node weights
  tbb::parallel_for(tbb::blocked_range<DNodeID>(0, c_n), [&](const auto &r) {
    for (DNodeID c_u = r.begin(); c_u < r.end(); ++c_u) {
      const auto first = buckets_index[c_u];
      const auto last = buckets_index[c_u + 1];

      for (std::size_t i = first; i < last; ++i) {
        const DNodeID u = buckets[i];
        c_node_weights[c_u] += graph.node_weight(u);
      }
    }
  });

  //
  // Send coarse node IDs to adjacent PEs
  //

  // Count messages
  scalable_vector<shm::parallel::IntegralAtomicWrapper<std::size_t>> num_messages_to_pe(size);
  graph.pfor_nodes([&](const DNodeID u) {
    for (const auto [e, v] : graph.neighbors(u)) {
      if (!graph.is_owned_node(v)) { num_messages_to_pe[graph.ghost_owner(v)] += 3; }
    }
  });

  scalable_vector<scalable_vector<std::int64_t>> send_buffer;
  for (const auto &num : num_messages_to_pe) { send_buffer.emplace_back(num); }

  // Prepare messages to ghost nodes
  graph.pfor_nodes([&](const DNodeID u) {
    for (const auto [e, v] : graph.neighbors(u)) {
      if (!graph.is_owned_node(v)) { // interface node
        const auto owner = graph.ghost_owner(v);
        const std::size_t pos = num_messages_to_pe[owner].fetch_sub(3);
        send_buffer[owner][pos - 3] = graph.global_node(u);
        send_buffer[owner][pos - 2] = first_node + mapping[u];
        send_buffer[owner][pos - 1] = c_node_weights[mapping[u]];
      }
    }
  });

  // Exchange messages
  scalable_vector<MPI_Request> requests;
  requests.reserve(size - 1);
  for (PEID pe = 0; pe < size; ++pe) {
    if (pe != rank) {
      requests.emplace_back();
      MPI_Isend(send_buffer[pe].data(), send_buffer[pe].size(), MPI_INT64_T, pe, 0, MPI_COMM_WORLD, &requests.back());
    }
  }

  scalable_vector<PEID> c_ghost_owner;
  scalable_vector<DNodeID> c_ghost_to_global;
  std::unordered_map<DNodeID, DNodeID> c_global_to_ghost;

  DNodeID c_next_ghost_node = c_n;

  for (PEID pe = 0; pe < size; ++pe) {
    if (pe != rank) {
      MPI_Status status{};
      MPI_Probe(pe, 0, MPI_COMM_WORLD, &status);
      int count = 0;
      MPI_Get_count(&status, MPI_INT64_T, &count);

      scalable_vector<int64_t> recv_messages(count);
      MPI_Recv(recv_messages.data(), count, MPI_INT64_T, pe, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      ASSERT(recv_messages.size() % 3 == 0); // we send triples of int64_t's

      // TODO parallelize? -> concurrent hash map
      for (std::size_t i = 0; i < recv_messages.size(); i += 3) {
        const DNodeID old_global_u = recv_messages[i];
        const DNodeID new_global_u = recv_messages[i + 1];
        const DNodeWeight new_weight = recv_messages[i + 2];

        const DNodeID old_local_u = graph.local_node(old_global_u);
        if (!c_global_to_ghost.contains(new_global_u)) {
          c_global_to_ghost[new_global_u] = c_next_ghost_node++;
          c_node_weights.push_back(new_weight);
          c_ghost_owner.push_back(pe);
          c_ghost_to_global.push_back(new_global_u);
        }
        mapping[old_local_u] = c_global_to_ghost[new_global_u];
      }
    }
  }
  MPI_Waitall(requests.size(), requests.data(), MPI_STATUS_IGNORE);

  DBG << V(c_n) << V(c_next_ghost_node);

  //
  // Remap coarse node IDs for ghost nodes to local coarse node IDs
  //

  tbb::enumerable_thread_specific<shm::RatingMap<DEdgeWeight>> collector{
      [&] { return shm::RatingMap<DEdgeWeight>(c_next_ghost_node); }};

  //
  // We build the coarse graph in multiple steps:
  // (1) During the first step, we compute
  //     - the node weight of each coarse node
  //     - the degree of each coarse node
  //     We can't build c_edges and c_edge_weights yet, because positioning edges in those arrays depends on c_nodes,
  //     which we only have after computing a prefix sum over all coarse node degrees
  //     Hence, we store edges and edge weights in unsorted auxiliary arrays during the first pass
  // (2) We finalize c_nodes arrays by computing a prefix sum over all coarse node degrees
  // (3) We copy coarse edges and coarse edge weights from the auxiliary arrays to c_edges and c_edge_weights
  //

  tbb::enumerable_thread_specific<LocalEdgeMemory> shared_edge_buffer;
  tbb::enumerable_thread_specific<std::vector<BufferNode>> shared_node_buffer;

  tbb::parallel_for(tbb::blocked_range<DNodeID>(0, c_n), [&](const auto &r) {
    auto &local_collector = collector.local();
    auto &local_edge_buffer = shared_edge_buffer.local();
    auto &local_node_buffer = shared_node_buffer.local();

    for (DNodeID c_u = r.begin(); c_u != r.end(); ++c_u) {
      std::size_t edge_buffer_position = local_edge_buffer.get_current_position();
      local_node_buffer.emplace_back(c_u, edge_buffer_position, &local_edge_buffer);

      const std::size_t first = buckets_index[c_u];
      const std::size_t last = buckets_index[c_u + 1];

      // build coarse graph
      auto collect_edges = [&](auto &map) {
        for (std::size_t i = first; i < last; ++i) {
          const DNodeID u = buckets[i];
          ASSERT(mapping[u] == c_u);

          // collect coarse edges
          for (const auto [e, v] : graph.neighbors(u)) {
            const DNodeID c_v = mapping[v];
            if (c_u != c_v) { map[c_v] += graph.edge_weight(e); }
          }
        }

        c_nodes[c_u + 1] = map.size(); // node degree (used to build c_nodes)

        // since we don't know the value of c_nodes[c_u] yet (so far, it only holds the nodes degree), we can't place the
        // edges of c_u in the c_edges and c_edge_weights arrays; hence, we store them in auxiliary arrays and note their
        // position in the auxiliary arrays
        for (const auto [c_v, weight] : map.entries()) { local_edge_buffer.push(c_v, weight); }
        map.clear();
      };

      // to select the right map, we compute a upper bound on the coarse node degree by summing the degree of all fine
      // nodes
      DEdgeID upper_bound_degree = 0;
      for (std::size_t i = first; i < last; ++i) {
        const DNodeID u = buckets[i];
        upper_bound_degree += graph.degree(u);
      }
      local_collector.update_upper_bound_size(upper_bound_degree);
      local_collector.run_with_map(collect_edges, collect_edges);
    }
  });

  shm::parallel::prefix_sum(c_nodes.begin(), c_nodes.end(), c_nodes.begin());

  ASSERT(c_nodes[0] == 0) << V(c_nodes);
  const DEdgeID c_m = c_nodes.back();

  //
  // Construct rest of the coarse graph: edges, edge weights
  //
  if (all_buffered_nodes.size() < c_n) { all_buffered_nodes.resize(c_n); }

  shm::parallel::IntegralAtomicWrapper<std::size_t> global_pos = 0;

  tbb::parallel_invoke(
      [&] {
        tbb::parallel_for(shared_edge_buffer.range(), [&](auto &r) {
          for (auto &buffer : r) {
            if (!buffer.current_chunk.empty()) { buffer.flush(); }
          }
        });
      },
      [&] {
        tbb::parallel_for(shared_node_buffer.range(), [&](const auto &r) {
          for (const auto &buffer : r) {
            const std::size_t local_pos = global_pos.fetch_add(buffer.size());
            std::copy(buffer.begin(), buffer.end(), all_buffered_nodes.begin() + local_pos);
          }
        });
      });

  scalable_vector<DNodeID> c_edges(c_m);
  scalable_vector<DEdgeWeight> c_edge_weights(c_m);

  // build coarse graph
  tbb::parallel_for(static_cast<DNodeID>(0), c_n, [&](const DNodeID i) {
    const auto &buffered_node = all_buffered_nodes[i];
    const auto *chunks = buffered_node.chunks;
    const DNodeID c_u = buffered_node.c_u;

    const DEdgeID c_u_degree = c_nodes[c_u + 1] - c_nodes[c_u];
    const DEdgeID first_target_index = c_nodes[c_u];
    const DEdgeID first_source_index = buffered_node.position;

    for (std::size_t j = 0; j < c_u_degree; ++j) {
      const auto to = first_target_index + j;
      const auto [c_v, weight] = chunks->get(first_source_index + j);
      c_edges[to] = c_v;
      c_edge_weights[to] = weight;
    }
  });

  // compute edge distribution
  DEdgeID last_edge;
  MPI_Scan(&c_m, &last_edge, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
  const DEdgeID first_edge = last_edge - c_m;

  scalable_vector<DEdgeID> c_edge_distribution(size + 1);
  c_edge_distribution[rank + 1] = last_edge;
  MPI_Allgather(&c_edge_distribution[rank + 1], 1, MPI_UINT64_T, c_edge_distribution.data() + 1, 1, MPI_UINT64_T,
                MPI_COMM_WORLD);
  const DEdgeID c_global_m = c_edge_distribution.back();

  DistributedGraph c_graph{c_global_n,
                           c_global_m,
                           c_next_ghost_node - c_n,
                           first_node,
                           first_edge,
                           std::move(c_node_distribution),
                           std::move(c_edge_distribution),
                           std::move(c_nodes),
                           std::move(c_edges),
                           std::move(c_node_weights),
                           std::move(c_edge_weights),
                           std::move(c_ghost_owner),
                           std::move(c_ghost_to_global),
                           std::move(c_global_to_ghost)};

  DBG << V(c_graph.n()) << V(c_graph.m()) << V(c_graph.ghost_n()) << V(c_graph.total_n()) << V(c_graph.global_n())
      << V(c_graph.global_m());

  return {std::move(c_graph), std::move(mapping), std::move(m_ctx)};
}

} // namespace dkaminpar::graph