/*******************************************************************************
 * @file:   local_graph_contraction.cc
 *
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 * @brief:
 ******************************************************************************/
#include "local_clustering_contraction.h"

#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>

#include "dkaminpar/mpi_graph.h"
#include "dkaminpar/mpi_wrapper.h"
#include "kaminpar/datastructure/rating_map.h"

namespace dkaminpar::coarsening {
using namespace contraction;

SET_DEBUG(true);

/*
 * Local cluster contraction
 */

Result contract_local_clustering(
    const DistributedGraph& graph, const scalable_vector<shm::parallel::IntegralAtomicWrapper<NodeID>>& clustering,
    MemoryContext m_ctx) {
    ASSERT(clustering.size() >= graph.n());

    MPI_Comm comm           = graph.communicator();
    const auto [size, rank] = mpi::get_comm_info(comm);

    auto& buckets_index      = m_ctx.buckets_index;
    auto& buckets            = m_ctx.buckets;
    auto& leader_mapping     = m_ctx.leader_mapping;
    auto& all_buffered_nodes = m_ctx.all_buffered_nodes;

    scalable_vector<NodeID> mapping(graph.total_n());
    if (leader_mapping.size() < graph.n()) {
        leader_mapping.resize(graph.n());
    }
    if (buckets.size() < graph.n()) {
        buckets.resize(graph.n());
    }

    //
    // Compute a mapping from the nodes of the current graph to the nodes of the coarse graph
    // I.e., node_mapping[node u] = coarse node c_u
    //

    // Set node_mapping[x] = 1 iff. there is a cluster with leader x
    graph.pfor_nodes([&](const NodeID u) { leader_mapping[u] = 0; });
    graph.pfor_nodes([&](const NodeID u) {
        ASSERT(clustering[u] < leader_mapping.size()) << V(clustering[u]) << V(leader_mapping.size());
        leader_mapping[clustering[u]].store(1, std::memory_order_relaxed);
    });

    // Compute prefix sum to get coarse node IDs (starting at 1!)
    shm::parallel::prefix_sum(leader_mapping.begin(), leader_mapping.begin() + graph.n(), leader_mapping.begin());
    const NodeID c_n = leader_mapping[graph.n() - 1]; // number of nodes in the coarse graph

    // Compute new node distribution, total number of coarse nodes
    const GlobalNodeID last_node  = mpi::scan(static_cast<GlobalNodeID>(c_n), MPI_SUM, comm);
    const GlobalNodeID first_node = last_node - c_n;

    scalable_vector<GlobalNodeID> c_node_distribution(size + 1);
    c_node_distribution[rank + 1] = last_node;
    mpi::allgather(&c_node_distribution[rank + 1], 1, c_node_distribution.data() + 1, 1, comm);

    // Assign coarse node ID to all nodes
    graph.pfor_nodes([&](const NodeID u) { mapping[u] = leader_mapping[clustering[u]]; });
    graph.pfor_nodes([&](const NodeID u) { --mapping[u]; });

    buckets_index.clear();
    buckets_index.resize(c_n + 1);

    //
    // Sort nodes into buckets: place all nodes belonging to coarse node i into the i-th bucket
    //
    // Count the number of nodes in each bucket, then compute the position of the bucket in the global buckets array
    // using a prefix sum, roughly 2/5-th of time on europe.osm with 2/3-th to 1/3-tel for loop to prefix sum
    graph.pfor_nodes([&](const NodeID u) { buckets_index[mapping[u]].fetch_add(1, std::memory_order_relaxed); });

    shm::parallel::prefix_sum(buckets_index.begin(), buckets_index.end(), buckets_index.begin());
    ASSERT(buckets_index.back() <= graph.n());

    // Sort nodes into   buckets, roughly 3/5-th of time on europe.osm
    graph.pfor_nodes([&](const NodeID u) {
        const std::size_t pos = buckets_index[mapping[u]].fetch_sub(1, std::memory_order_relaxed) - 1;
        buckets[pos]          = u;
    });

    //
    // Build nodes array of the coarse graph
    // - firstly, we count the degree of each coarse node
    // - secondly, we obtain the nodes array using a prefix sum
    //
    scalable_vector<EdgeID>     c_nodes(c_n + 1);
    scalable_vector<NodeWeight> c_node_weights(c_n);

    // Build coarse node weights
    tbb::parallel_for<NodeID>(0, c_n, [&](const NodeID c_u) {
        const auto first = buckets_index[c_u];
        const auto last  = buckets_index[c_u + 1];

        for (std::size_t i = first; i < last; ++i) {
            const NodeID u = buckets[i];
            c_node_weights[c_u] += graph.node_weight(u);
        }
    });

    //
    // Sparse all-to-all for building node mapping for ghost nodes
    //

    struct CoarseGhostNode {
        GlobalNodeID old_global_node;
        GlobalNodeID new_global_node;
        NodeWeight   coarse_weight;
    };

    scalable_vector<PEID>                    c_ghost_owner;
    scalable_vector<GlobalNodeID>            c_ghost_to_global;
    std::unordered_map<GlobalNodeID, NodeID> c_global_to_ghost;
    NodeID                                   c_next_ghost_node = c_n;

    mpi::graph::sparse_alltoall_interface_to_pe<CoarseGhostNode>(
        graph,
        [&](const NodeID u) -> CoarseGhostNode {
            ASSERT(u < mapping.size());
            ASSERT(mapping[u] < c_node_weights.size());

            return {
                .old_global_node = graph.local_to_global_node(u),
                .new_global_node = first_node + mapping[u],
                .coarse_weight   = c_node_weights[mapping[u]],
            };
        },
        [&](const auto recv_buffer, const PEID pe) { // TODO parallelize
            for (const auto [old_global_u, new_global_u, new_weight]: recv_buffer) {
                const NodeID old_local_u = graph.global_to_local_node(old_global_u);
                if (!c_global_to_ghost.contains(new_global_u)) {
                    c_global_to_ghost[new_global_u] = c_next_ghost_node++;
                    c_node_weights.push_back(new_weight);
                    c_ghost_owner.push_back(pe);
                    c_ghost_to_global.push_back(new_global_u);
                }
                mapping[old_local_u] = c_global_to_ghost[new_global_u];
            }
        });

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
    using Map = shm::RatingMap<EdgeWeight, shm::FastResetArray<EdgeWeight, NodeID>>;
    tbb::enumerable_thread_specific<Map>   collector_ets{[&] {
        return Map(c_next_ghost_node);
    }};
    shm::NavigableLinkedList<NodeID, Edge> edge_buffer_ets;

    tbb::parallel_for(tbb::blocked_range<NodeID>(0, c_n), [&](const auto& r) {
        auto& local_collector   = collector_ets.local();
        auto& local_edge_buffer = edge_buffer_ets.local();

        for (NodeID c_u = r.begin(); c_u != r.end(); ++c_u) {
            local_edge_buffer.mark(c_u);

            const std::size_t first = buckets_index[c_u];
            const std::size_t last  = buckets_index[c_u + 1];

            // build coarse graph
            auto collect_edges = [&](auto& map) {
                for (std::size_t i = first; i < last; ++i) {
                    const NodeID u = buckets[i];
                    ASSERT(mapping[u] == c_u);

                    // collect coarse edges
                    for (const auto [e, v]: graph.neighbors(u)) {
                        const NodeID c_v = mapping[v];
                        if (c_u != c_v) {
                            map[c_v] += graph.edge_weight(e);
                        }
                    }
                }

                c_nodes[c_u + 1] = map.size(); // node degree (used to build c_nodes)

                // since we don't know the value of c_nodes[c_u] yet (so far, it only holds the nodes degree), we can't
                // place the edges of c_u in the c_edges and c_edge_weights arrays; hence, we store them in auxiliary
                // arrays and note their position in the auxiliary arrays
                for (const auto [c_v, weight]: map.entries()) {
                    local_edge_buffer.push_back({c_v, weight});
                }
                map.clear();
            };

            // to select the right map, we compute an upper bound on the coarse node degree by summing the degree of all
            // fine nodes
            EdgeID upper_bound_degree = 0;
            for (std::size_t i = first; i < last; ++i) {
                const NodeID u = buckets[i];
                upper_bound_degree += graph.degree(u);
            }
            local_collector.update_upper_bound_size(upper_bound_degree);
            local_collector.run_with_map(collect_edges, collect_edges);
        }
    });

    shm::parallel::prefix_sum(c_nodes.begin(), c_nodes.end(), c_nodes.begin());

    ASSERT(c_nodes[0] == 0) << V(c_nodes);
    const EdgeID c_m = c_nodes.back();

    //
    // Construct rest of the coarse graph: edges, edge weights
    //
    all_buffered_nodes = shm::ts_navigable_list::combine<NodeID, Edge>(edge_buffer_ets, std::move(all_buffered_nodes));

    scalable_vector<NodeID>     c_edges(c_m);
    scalable_vector<EdgeWeight> c_edge_weights(c_m);

    // build coarse graph
    tbb::parallel_for(static_cast<NodeID>(0), c_n, [&](const NodeID i) {
        const auto&  marker = all_buffered_nodes[i];
        const auto*  list   = marker.local_list;
        const NodeID c_u    = marker.key;

        const EdgeID c_u_degree         = c_nodes[c_u + 1] - c_nodes[c_u];
        const EdgeID first_target_index = c_nodes[c_u];
        const EdgeID first_source_index = marker.position;

        for (std::size_t j = 0; j < c_u_degree; ++j) {
            const auto to            = first_target_index + j;
            const auto [c_v, weight] = list->get(first_source_index + j);
            c_edges[to]              = c_v;
            c_edge_weights[to]       = weight;
        }
    });

    // compute edge distribution
    const GlobalEdgeID            last_edge = mpi::scan(static_cast<GlobalEdgeID>(c_m), MPI_SUM, comm);
    scalable_vector<GlobalEdgeID> c_edge_distribution(size + 1);
    c_edge_distribution[rank + 1] = last_edge;
    mpi::allgather(&c_edge_distribution[rank + 1], 1, c_edge_distribution.data() + 1, 1, comm);

    // TODO
    growt::StaticGhostNodeMapping static_mapping(c_global_to_ghost.size());
    for (const auto& [key, value]: c_global_to_ghost) {
        static_mapping.insert(key, value);
    }

    DistributedGraph c_graph{
        std::move(c_node_distribution),
        std::move(c_edge_distribution),
        std::move(c_nodes),
        std::move(c_edges),
        std::move(c_node_weights),
        std::move(c_edge_weights),
        std::move(c_ghost_owner),
        std::move(c_ghost_to_global),
        std::move(static_mapping),
        graph.communicator()};

    DBG << V(c_graph.n()) << V(c_graph.m()) << V(c_graph.ghost_n()) << V(c_graph.total_n()) << V(c_graph.global_n())
        << V(c_graph.global_m());

    return {std::move(c_graph), std::move(mapping), std::move(m_ctx)};
}
} // namespace dkaminpar::coarsening