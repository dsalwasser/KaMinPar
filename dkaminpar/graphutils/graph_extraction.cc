/*******************************************************************************
 * @file:   graph_extraction.cc
 *
 * @author: Daniel Seemaier
 * @date:   28.04.2022
 * @brief:  Distributes block-induced subgraphs of a partitioned graph across
 * PEs.
 ******************************************************************************/
#include "dkaminpar/graphutils/graph_extraction.h"

#include "dkaminpar/datastructure/distributed_graph.h"
#include "dkaminpar/mpi_graph.h"
#include "dkaminpar/utils/math.h"
#include "dkaminpar/utils/vector_ets.h"
#include "kaminpar/parallel/algorithm.h"
#include "mpi_wrapper.h"

#include <functional>
#include <mpi.h>

namespace dkaminpar::graph {
SET_DEBUG(true);

namespace {
PEID compute_block_owner(const BlockID b, const BlockID k, const PEID num_pes) {
    return static_cast<PEID>(math::compute_local_range_rank<BlockID>(k, static_cast<BlockID>(num_pes), b));
}

auto count_block_induced_subgraph_sizes(const DistributedPartitionedGraph& p_graph) {
    parallel::vector_ets<NodeID> num_nodes_per_block_ets(p_graph.k());
    parallel::vector_ets<EdgeID> num_edges_per_block_ets(p_graph.k());

    tbb::parallel_for(tbb::blocked_range<NodeID>(0, p_graph.n()), [&](const auto r) {
        auto& num_nodes_per_block = num_nodes_per_block_ets.local();
        auto& num_edges_per_block = num_edges_per_block_ets.local();
        for (NodeID u = r.begin(); u != r.end(); ++u) {
            const BlockID u_block = p_graph.block(u);
            ++num_nodes_per_block[u_block];
            for (const auto [e, v]: p_graph.neighbors(u)) {
                if (u_block == p_graph.block(v)) {
                    ++num_edges_per_block[u_block];
                }
            }
        }
    });

    return std::make_pair(num_nodes_per_block_ets.combine(std::plus{}), num_edges_per_block_ets.combine(std::plus{}));
}
} // namespace

// Build a local block-induced subgraph for each block of the graph partition.
ExtractedSubgraphs
extract_local_block_induced_subgraphs(const DistributedPartitionedGraph& p_graph, ExtractedSubgraphs memory) {
    auto [num_nodes_per_block, num_edges_per_block] = count_block_induced_subgraph_sizes(p_graph);
    const EdgeID num_internal_edges = std::accumulate(num_edges_per_block.begin(), num_edges_per_block.end(), 0);
    DBG << V(num_nodes_per_block) << V(num_edges_per_block);

    auto& shared_nodes          = memory.shared_nodes;
    auto& shared_node_weights   = memory.shared_node_weights;
    auto& shared_edges          = memory.shared_edges;
    auto& shared_edge_weights   = memory.shared_edge_weights;
    auto& nodes_offset          = memory.nodes_offset;
    auto& edges_offset          = memory.edges_offset;
    auto& mapping               = memory.mapping;
    auto  next_node_in_subgraph = std::vector<parallel::Atomic<NodeID>>();

    // Allocate memory
    {
        SCOPED_TIMER("Allocation", TIMER_DETAIL);

        const std::size_t min_nodes_size   = p_graph.n();
        const std::size_t min_edges_size   = num_internal_edges;
        const std::size_t min_offset_size  = p_graph.k() + 1;
        const std::size_t min_mapping_size = p_graph.total_n();

        LIGHT_ASSERT(shared_nodes.size() == shared_node_weights.size());
        LIGHT_ASSERT(shared_edges.size() == shared_edge_weights.size());
        LIGHT_ASSERT(nodes_offset.size() == edges_offset.size());

        if (shared_nodes.size() < min_nodes_size) {
            shared_nodes.resize(min_nodes_size);
            shared_node_weights.resize(min_nodes_size);
        }
        if (shared_edges.size() < min_edges_size) {
            shared_edges.resize(min_edges_size);
            shared_edge_weights.resize(min_edges_size);
        }
        if (nodes_offset.size() < min_offset_size) {
            nodes_offset.resize(min_offset_size);
            edges_offset.resize(min_offset_size);
        }
        if (mapping.size() < min_mapping_size) {
            mapping.resize(min_mapping_size);
        }

        next_node_in_subgraph.resize(p_graph.k());
    }

    // Compute of graphs in shared_* arrays
    {
        SCOPED_TIMER("Compute subgraph offsets", TIMER_DETAIL);

        parallel::prefix_sum(num_nodes_per_block.begin(), num_nodes_per_block.end(), nodes_offset.begin() + 1);
        parallel::prefix_sum(num_edges_per_block.begin(), num_edges_per_block.end(), edges_offset.begin() + 1);
        DBG << V(nodes_offset) << V(edges_offset);
    }

    // Compute node ID offset of local subgraph in global subgraphs
    std::vector<NodeID> global_node_offset(p_graph.k());
    mpi::exscan(num_nodes_per_block.data(), global_node_offset.data(), p_graph.k(), MPI_SUM, p_graph.communicator());

    // Build mapping from node IDs in p_graph to node IDs in the extracted subgraph
    {
        SCOPED_TIMER("Build node mapping", TIMER_DETAIL);

        // @todo bottleneck for scalibility
        p_graph.pfor_nodes([&](const NodeID u) {
            const BlockID b               = p_graph.block(u);
            const NodeID  pos_in_subgraph = next_node_in_subgraph[b]++;
            const NodeID  pos             = nodes_offset[b] + pos_in_subgraph;
            shared_nodes[pos] = u;
            mapping[u]        = global_node_offset[b] + pos_in_subgraph;
        });
    }

    // Build mapping from local extract subgraph to global extracted subgraph for ghost nodes
    std::vector<NodeID> global_ghost_node_mapping(p_graph.ghost_n());

    {
        SCOPED_TIMER("Exchange ghost node mapping", TIMER_DETAIL);

        struct NodeToMappedNode {
            GlobalNodeID global_node;
            NodeID       mapped_node;
        };

        mpi::graph::sparse_alltoall_interface_to_pe<NodeToMappedNode>(
            p_graph.graph(),
            [&](const NodeID u) {
                return NodeToMappedNode{p_graph.local_to_global_node(u), mapping[u]};
            },
            [&](const auto buffer) {
                tbb::parallel_for<std::size_t>(0, buffer.size(), [&](const std::size_t i) {
                    const auto& [global_node, mapped_node] = buffer[i];
                    const NodeID local_node                = p_graph.global_to_local_node(global_node);
                    mapping[local_node]                    = mapped_node;
                });
            });
    }

    // Extract the subgraphs
    {
        SCOPED_TIMER("Extract subgraphs", TIMER_DETAIL);

        tbb::parallel_for<BlockID>(0, p_graph.k(), [&](const BlockID b) {
            const NodeID n0 = nodes_offset[b];
            const EdgeID e0 = edges_offset[b];
            EdgeID       e  = 0;

            // u, v, e = IDs in extracted subgraph
            // u_prime, v_prime, e_prime = IDs in p_graph
            for (NodeID u = 0; u < next_node_in_subgraph[b]; ++u) {
                const NodeID pos     = n0 + u;
                const NodeID u_prime = shared_nodes[pos];

                for (const auto [e_prime, v_prime]: p_graph.neighbors(u_prime)) {
                    if (p_graph.block(v_prime) != b) {
                        continue;
                    }

                    shared_edge_weights[e0 + e] = p_graph.edge_weight(e_prime);
                    shared_edges[e0 + e]        = mapping[v_prime];
                    ++e;
                }

                shared_nodes[pos]        = e;
                shared_node_weights[pos] = p_graph.node_weight(u_prime);
            }
        });
    }

    return memory;
}

void gather_block_induced_subgraphs(const DistributedPartitionedGraph& p_graph, ExtractedSubgraphs memory) {
    const PEID size = mpi::get_comm_size(p_graph.communicator());
    const PEID rank = mpi::get_comm_rank(p_graph.communicator());
    ALWAYS_ASSERT(p_graph.k() % size == 0) << "k must be a multiple of #PEs";
    const BlockID blocks_per_pe = p_graph.k() / size;

    // Communicate recvcounts
    struct SubgraphSize {
        NodeID n;
        EdgeID m;

        SubgraphSize operator+(const SubgraphSize other) {
            return {n + other.n, m + other.m};
        }
    };

    struct GraphSize {
        NodeID n;
        EdgeID m;
    };
    std::vector<GraphSize> recv_subgraph_sizes(p_graph.k());

    {
        SCOPED_TIMER("Alltoall recvcounts", TIMER_DETAIL);

        START_TIMER("Compute counts", TIMER_DETAIL);
        std::vector<GraphSize> send_subgraph_sizes(p_graph.k());
        p_graph.pfor_blocks([&](const BlockID b) {
            send_subgraph_sizes[b].n = memory.nodes_offset[b + 1] - memory.nodes_offset[b];
            send_subgraph_sizes[b].m = memory.edges_offset[b + 1] - memory.edges_offset[b];
        });
        STOP_TIMER(TIMER_DETAIL);

        START_TIMER("MPI_Alltoall", TIMER_DETAIL);
        mpi::alltoall(send_subgraph_sizes.data(), blocks_per_pe, recv_subgraph_sizes.data(), blocks_per_pe);
        STOP_TIMER(TIMER_DETAIL);
    }

    // Exchange subgraphs
    std::vector<EdgeID>     shared_nodes;
    std::vector<NodeWeight> shared_node_weights;
    std::vector<NodeID>     shared_edges;
    std::vector<EdgeWeight> shared_edge_weights;

    {
        SCOPED_TIMER("Alltoallv block-induced subgraphs", TIMER_DETAIL);

        START_TIMER("Allocation", TIMER_DETAIL);
        std::vector<int> sendcounts_nodes(size);
        std::vector<int> sendcounts_edges(size);
        std::vector<int> sdispls_nodes(size);
        std::vector<int> sdispls_edges(size);
        std::vector<int> recvcounts_nodes(size);
        std::vector<int> recvcounts_edges(size);
        std::vector<int> rdispls_nodes(size);
        std::vector<int> rdispls_edges(size);
        STOP_TIMER(TIMER_DETAIL);

        START_TIMER("Compute counts and displs", TIMER_DETAIL);
        tbb::parallel_for<PEID>(0, size, [&](const PEID pe) {
            const BlockID first_block_on_pe         = pe * blocks_per_pe;
            const BlockID first_invalid_block_on_pe = (pe + 1) * blocks_per_pe;

            sendcounts_nodes[pe] =
                memory.nodes_offset[first_invalid_block_on_pe] - memory.nodes_offset[first_block_on_pe];
            sendcounts_edges[pe] =
                memory.edges_offset[first_invalid_block_on_pe] - memory.edges_offset[first_block_on_pe];

            for (BlockID b = first_block_on_pe; b < first_invalid_block_on_pe; ++b) {
                recvcounts_nodes[b] += recv_subgraph_sizes[b].n;
                recvcounts_edges[b] += recv_subgraph_sizes[b].m;
            }
        });
        STOP_TIMER(TIMER_DETAIL);

        START_TIMER("MPI_Alltoallv", TIMER_DETAIL);
        mpi::alltoallv(
            memory.shared_nodes.data(), sendcounts_nodes.data(), sdispls_nodes.data(), shared_nodes.data(),
            recvcounts_nodes.data(), rdispls_nodes.data(), p_graph.communicator());
        mpi::alltoallv(
            memory.shared_node_weights.data(), sendcounts_nodes.data(), sdispls_nodes.data(),
            shared_node_weights.data(), recvcounts_nodes.data(), rdispls_nodes.data(), p_graph.communicator());
        mpi::alltoallv(
            memory.shared_edges.data(), sendcounts_edges.data(), sdispls_edges.data(), shared_edges.data(),
            recvcounts_edges.data(), rdispls_edges.data(), p_graph.communicator());
        mpi::alltoallv(
            memory.shared_edge_weights.data(), sendcounts_edges.data(), sdispls_edges.data(),
            shared_edge_weights.data(), recvcounts_edges.data(), rdispls_edges.data(), p_graph.communicator());
        STOP_TIMER(TIMER_DETAIL);
    }

    // Construct subgraphs
    std::vector<shm::StaticArray<EdgeID>>     subgraph_nodes(blocks_per_pe);
    std::vector<shm::StaticArray<NodeWeight>> subgraph_node_weights(blocks_per_pe);
    std::vector<shm::StaticArray<NodeID>>     subgraph_edges(blocks_per_pe);
    std::vector<shm::StaticArray<EdgeWeight>> subgraph_edge_weights(blocks_per_pe);
    std::vector<shm::Graph>                   subgraphs(blocks_per_pe);

    {
        SCOPED_TIMER("Construct subgraphs", TIMER_DETAIL);

        tbb::parallel_for<BlockID>(0, blocks_per_pe, [&](const BlockID b) {
            NodeID n = 0;
            EdgeID m = 0;
            for (PEID pe = 0; pe < size; ++pe) {
                const std::size_t i = b + pe * blocks_per_pe;
                n += recv_subgraph_sizes[i].n;
                m += recv_subgraph_sizes[i].m;
            }

            // Allocate memory for subgraph
            subgraph_nodes[b].resize(n + 1);
            subgraph_node_weights[b].resize(n);
            subgraph_edges[b].resize(m);
            subgraph_edge_weights[b].resize(m);

            // Copy subgraph to memory
            // @todo better approach might be to compute a prefix sum on recv_subgraph_sizes
            for (PEID pe = 0; pe < size; ++pe) {
            }
        });
    }
}

std::vector<DistributedGraph> distribute_block_induced_subgraphs(const DistributedPartitionedGraph& p_graph) {
    return {};
}
} // namespace dkaminpar::graph