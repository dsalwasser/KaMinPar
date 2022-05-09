/*******************************************************************************
 * @file:   distributed_graph.h
 *
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 * @brief:  Static distributed graph data structure.
 ******************************************************************************/
#pragma once

#include <vector>

#include <tbb/parallel_for.h>

#include "dkaminpar/definitions.h"
#include "dkaminpar/growt.h"
#include "dkaminpar/mpi_wrapper.h"
#include "kaminpar/datastructure/graph.h"
#include "kaminpar/datastructure/marker.h"
#include "kaminpar/parallel/algorithm.h"
#include "kaminpar/utils/ranges.h"

namespace dkaminpar {
namespace graph {
[[nodiscard]] inline growt::StaticGhostNodeMapping
build_static_ghost_node_mapping(std::unordered_map<GlobalNodeID, NodeID> global_to_ghost) {
    growt::StaticGhostNodeMapping static_mapping(global_to_ghost.size());
    for (const auto& [key, value]: global_to_ghost) {
        static_mapping.insert(key + 1, value); // 0 cannot be used as a key in growt hash tables
    }
    return static_mapping;
}
} // namespace graph

class DistributedGraph {
    SET_DEBUG(true);

public:
    using NodeID           = ::dkaminpar::NodeID;
    using GlobalNodeID     = ::dkaminpar::GlobalNodeID;
    using NodeWeight       = ::dkaminpar::NodeWeight;
    using GlobalNodeWeight = ::dkaminpar::GlobalNodeWeight;
    using EdgeID           = ::dkaminpar::EdgeID;
    using GlobalEdgeID     = ::dkaminpar::GlobalEdgeID;
    using EdgeWeight       = ::dkaminpar::EdgeWeight;
    using GlobalEdgeWeight = ::dkaminpar::GlobalEdgeWeight;

    DistributedGraph() = default;

    DistributedGraph(
        scalable_vector<GlobalNodeID> node_distribution, scalable_vector<GlobalEdgeID> edge_distribution,
        scalable_vector<EdgeID> nodes, scalable_vector<NodeID> edges, scalable_vector<NodeWeight> node_weights,
        scalable_vector<EdgeWeight> edge_weights, scalable_vector<PEID> ghost_owner,
        scalable_vector<GlobalNodeID> ghost_to_global, std::unordered_map<GlobalNodeID, NodeID> global_to_ghost,
        const bool sorted, MPI_Comm comm)
        : DistributedGraph{
            std::move(node_distribution),
            std::move(edge_distribution),
            std::move(nodes),
            std::move(edges),
            std::move(node_weights),
            std::move(edge_weights),
            std::move(ghost_owner),
            std::move(ghost_to_global),
            graph::build_static_ghost_node_mapping(std::move(global_to_ghost)),
            sorted,
            comm} {}

    DistributedGraph(
        scalable_vector<GlobalNodeID> node_distribution, scalable_vector<GlobalEdgeID> edge_distribution,
        scalable_vector<EdgeID> nodes, scalable_vector<NodeID> edges, scalable_vector<PEID> ghost_owner,
        scalable_vector<GlobalNodeID> ghost_to_global, growt::StaticGhostNodeMapping global_to_ghost, const bool sorted,
        MPI_Comm comm)
        : DistributedGraph{std::move(node_distribution),
                           std::move(edge_distribution),
                           std::move(nodes),
                           std::move(edges),
                           {},
                           {},
                           std::move(ghost_owner),
                           std::move(ghost_to_global),
                           std::move(global_to_ghost),
                           sorted,
                           comm} {}

    DistributedGraph(
        scalable_vector<GlobalNodeID> node_distribution, scalable_vector<GlobalEdgeID> edge_distribution,
        scalable_vector<EdgeID> nodes, scalable_vector<NodeID> edges, scalable_vector<NodeWeight> node_weights,
        scalable_vector<EdgeWeight> edge_weights, scalable_vector<PEID> ghost_owner,
        scalable_vector<GlobalNodeID> ghost_to_global, growt::StaticGhostNodeMapping global_to_ghost, const bool sorted,
        MPI_Comm comm)
        : _node_distribution{std::move(node_distribution)},
          _edge_distribution{std::move(edge_distribution)},
          _nodes{std::move(nodes)},
          _edges{std::move(edges)},
          _node_weights{std::move(node_weights)},
          _edge_weights{std::move(edge_weights)},
          _ghost_owner{std::move(ghost_owner)},
          _ghost_to_global{std::move(ghost_to_global)},
          _global_to_ghost{std::move(global_to_ghost)},
          _sorted{sorted},
          _communicator{comm} {
        PEID rank;
        MPI_Comm_rank(communicator(), &rank);

        _n        = _nodes.size() - 1;
        _m        = _edges.size();
        _ghost_n  = _ghost_to_global.size();
        _offset_n = _node_distribution[rank];
        _offset_m = _edge_distribution[rank];
        _global_n = _node_distribution.back();
        _global_m = _edge_distribution.back();

        init_total_node_weight();
        init_communication_metrics();
        init_degree_buckets();
    }

    DistributedGraph(const DistributedGraph&) = delete;
    DistributedGraph& operator=(const DistributedGraph&) = delete;
    DistributedGraph(DistributedGraph&&) noexcept        = default;
    DistributedGraph& operator=(DistributedGraph&&) noexcept = default;

    // Graph size
    [[nodiscard]] inline GlobalNodeID global_n() const {
        return _global_n;
    }
    [[nodiscard]] inline GlobalEdgeID global_m() const {
        return _global_m;
    }

    [[nodiscard]] inline NodeID n() const {
        return _n;
    }
    [[nodiscard]] inline NodeID n(const PEID pe) const {
        KASSERT(pe < static_cast<PEID>(_node_distribution.size()));
        return _node_distribution[pe + 1] - _node_distribution[pe];
    }
    [[nodiscard]] inline NodeID ghost_n() const {
        return _ghost_n;
    }
    [[nodiscard]] inline NodeID total_n() const {
        return ghost_n() + n();
    }

    [[nodiscard]] inline EdgeID m() const {
        return _m;
    }
    [[nodiscard]] inline EdgeID m(const PEID pe) const {
        KASSERT(pe < static_cast<PEID>(_edge_distribution.size()));
        return _edge_distribution[pe + 1] - _edge_distribution[pe];
    }

    [[nodiscard]] inline GlobalNodeID offset_n() const {
        return _offset_n;
    }
    [[nodiscard]] inline GlobalNodeID offset_n(const PEID pe) const {
        return _node_distribution[pe];
    }
    [[nodiscard]] inline GlobalEdgeID offset_m() const {
        return _offset_m;
    }
    [[nodiscard]] inline GlobalEdgeID offset_m(const PEID pe) const {
        return _edge_distribution[pe];
    }

    [[nodiscard]] inline bool is_node_weighted() const {
        return !_node_weights.empty();
    }
    [[nodiscard]] inline bool is_edge_weighted() const {
        return !_edge_weights.empty();
    }

    [[nodiscard]] inline NodeWeight total_node_weight() const {
        return _total_node_weight;
    }
    [[nodiscard]] inline GlobalNodeWeight global_total_node_weight() const {
        return _global_total_node_weight;
    }
    [[nodiscard]] inline NodeWeight max_node_weight() const {
        return _max_node_weight;
    }
    [[nodiscard]] inline GlobalNodeWeight global_max_node_weight() const {
        return _global_max_node_weight;
    }

    [[nodiscard]] inline bool is_owned_global_node(const GlobalNodeID global_u) const {
        return (offset_n() <= global_u && global_u < offset_n() + n());
    }

    [[nodiscard]] inline bool contains_global_node(const GlobalNodeID global_u) const {
        return is_owned_global_node(global_u)                                      // owned node
               || (_global_to_ghost.find(global_u + 1) != _global_to_ghost.end()); // ghost node
    }

    [[nodiscard]] inline bool contains_local_node(const NodeID local_u) const {
        return local_u < total_n();
    }

    // Node type
    [[nodiscard]] inline bool is_ghost_node(const NodeID u) const {
        KASSERT(u < total_n());
        return u >= n();
    }
    [[nodiscard]] inline bool is_owned_node(const NodeID u) const {
        KASSERT(u < total_n());
        return u < n();
    }

    // Distributed info
    [[nodiscard]] inline PEID ghost_owner(const NodeID u) const {
        KASSERT(is_ghost_node(u));
        KASSERT(u - n() < _ghost_owner.size());
        KASSERT(_ghost_owner[u - n()] >= 0u);
        KASSERT(_ghost_owner[u - n()] < mpi::get_comm_size(communicator()));
        return _ghost_owner[u - n()];
    }

    [[nodiscard]] inline GlobalNodeID local_to_global_node(const NodeID local_u) const {
        KASSERT(contains_local_node(local_u));
        return is_owned_node(local_u) ? _offset_n + local_u : _ghost_to_global[local_u - n()];
    }

    [[nodiscard]] inline NodeID global_to_local_node(const GlobalNodeID global_u) const {
        KASSERT(contains_global_node(global_u));

        if (offset_n() <= global_u && global_u < offset_n() + n()) {
            return global_u - offset_n();
        } else {
            KASSERT(_global_to_ghost.find(global_u + 1) != _global_to_ghost.end());
            return (*_global_to_ghost.find(global_u + 1)).second;
        }
    }

    // Access methods
    [[nodiscard]] inline NodeWeight node_weight(const NodeID u) const {
        KASSERT(u < total_n());
        KASSERT((!is_node_weighted() || u < _node_weights.size()));
        return is_node_weighted() ? _node_weights[u] : 1;
    }

    [[nodiscard]] inline const auto& node_weights() const {
        return _node_weights;
    }

    // convenient to have this for ghost nodes
    void set_ghost_node_weight(const NodeID ghost_node, const NodeWeight weight) {
        KASSERT(is_ghost_node(ghost_node));
        KASSERT(is_node_weighted());
        _node_weights[ghost_node] = weight;
    }

    [[nodiscard]] inline EdgeWeight edge_weight(const EdgeID e) const {
        KASSERT(e < m());
        KASSERT((!is_edge_weighted() || e < _edge_weights.size()));
        return is_edge_weighted() ? _edge_weights[e] : 1;
    }

    [[nodiscard]] inline const auto& edge_weights() const {
        return _edge_weights;
    }

    // Graph structure
    [[nodiscard]] inline EdgeID first_edge(const NodeID u) const {
        KASSERT(u < n());
        return _nodes[u];
    }

    [[nodiscard]] inline EdgeID first_invalid_edge(const NodeID u) const {
        KASSERT(u < n());
        return _nodes[u + 1];
    }

    [[nodiscard]] inline NodeID edge_target(const EdgeID e) const {
        KASSERT(e < m());
        return _edges[e];
    }

    [[nodiscard]] inline NodeID degree(const NodeID u) const {
        KASSERT(is_owned_node(u));
        return _nodes[u + 1] - _nodes[u];
    }

    [[nodiscard]] const auto& node_distribution() const {
        return _node_distribution;
    }

    [[nodiscard]] GlobalNodeID node_distribution(const PEID pe) const {
        KASSERT(static_cast<std::size_t>(pe) < _node_distribution.size());
        return _node_distribution[pe];
    }

    PEID find_owner_of_global_node(const GlobalNodeID u) const {
        KASSERT(u < global_n());
        auto it = std::upper_bound(_node_distribution.begin() + 1, _node_distribution.end(), u);
        KASSERT(it != _node_distribution.end());
        return static_cast<PEID>(std::distance(_node_distribution.begin(), it) - 1);
    }

    [[nodiscard]] const auto& edge_distribution() const {
        return _edge_distribution;
    }

    [[nodiscard]] GlobalEdgeID edge_distribution(const PEID pe) const {
        KASSERT(static_cast<std::size_t>(pe) < _edge_distribution.size());
        return _edge_distribution[pe];
    }

    [[nodiscard]] const auto& raw_nodes() const {
        return _nodes;
    }
    [[nodiscard]] const auto& raw_node_weights() const {
        return _node_weights;
    }
    [[nodiscard]] const auto& raw_edges() const {
        return _edges;
    }
    [[nodiscard]] const auto& raw_edge_weights() const {
        return _edge_weights;
    }

    // Parallel iteration
    template <typename Lambda>
    inline void pfor_nodes(const NodeID from, const NodeID to, Lambda&& l) const {
        tbb::parallel_for(from, to, std::forward<Lambda&&>(l));
    }

    template <typename Lambda>
    inline void pfor_nodes_range(const NodeID from, const NodeID to, Lambda&& l) const {
        tbb::parallel_for(tbb::blocked_range<NodeID>(from, to), std::forward<Lambda&&>(l));
    }

    template <typename Lambda>
    inline void pfor_nodes(Lambda&& l) const {
        pfor_nodes(0, n(), std::forward<Lambda&&>(l));
    }

    template <typename Lambda>
    inline void pfor_all_nodes(Lambda&& l) const {
        pfor_nodes(0, total_n(), std::forward<Lambda&&>(l));
    }

    template <typename Lambda>
    inline void pfor_nodes_range(Lambda&& l) const {
        pfor_nodes_range(0, n(), std::forward<Lambda&&>(l));
    }

    template <typename Lambda>
    inline void pfor_all_nodes_range(Lambda&& l) const {
        pfor_nodes_range(0, total_n(), std::forward<Lambda&&>(l));
    }

    template <typename Lambda>
    inline void pfor_edges(Lambda&& l) const {
        tbb::parallel_for(static_cast<EdgeID>(0), m(), std::forward<Lambda&&>(l));
    }

    // Iterators for nodes / edges
    [[nodiscard]] inline auto nodes() const {
        return shm::IotaRange(static_cast<NodeID>(0), n());
    }
    [[nodiscard]] inline auto ghost_nodes() const {
        return shm::IotaRange(n(), total_n());
    }
    [[nodiscard]] inline auto all_nodes() const {
        return shm::IotaRange(static_cast<NodeID>(0), total_n());
    }
    [[nodiscard]] inline auto edges() const {
        return shm::IotaRange(static_cast<EdgeID>(0), m());
    }
    [[nodiscard]] inline auto incident_edges(const NodeID u) const {
        return shm::IotaRange(_nodes[u], _nodes[u + 1]);
    }

    [[nodiscard]] inline auto adjacent_nodes(const NodeID u) const {
        return shm::TransformedIotaRange(
            _nodes[u], _nodes[u + 1], [this](const EdgeID e) { return this->edge_target(e); });
    }

    [[nodiscard]] inline auto neighbors(const NodeID u) const {
        return shm::TransformedIotaRange(
            _nodes[u], _nodes[u + 1], [this](const EdgeID e) { return std::make_pair(e, this->edge_target(e)); });
    }

    // Degree buckets
    [[nodiscard]] inline std::size_t bucket_size(const std::size_t bucket) const {
        return _buckets[bucket + 1] - _buckets[bucket];
    }
    [[nodiscard]] inline NodeID first_node_in_bucket(const std::size_t bucket) const {
        return _buckets[bucket];
    }
    [[nodiscard]] inline NodeID first_invalid_node_in_bucket(const std::size_t bucket) const {
        return first_node_in_bucket(bucket + 1);
    }
    [[nodiscard]] inline std::size_t number_of_buckets() const {
        return _number_of_buckets;
    }
    [[nodiscard]] inline bool sorted() const {
        return _sorted;
    }

    // Cached inter-PE metrics
    [[nodiscard]] inline EdgeID edge_cut_to_pe(const PEID pe) const {
        KASSERT(static_cast<std::size_t>(pe) < _edge_cut_to_pe.size());
        return _edge_cut_to_pe[pe];
    }

    [[nodiscard]] inline EdgeID comm_vol_to_pe(const PEID pe) const {
        KASSERT(static_cast<std::size_t>(pe) < _comm_vol_to_pe.size());
        return _comm_vol_to_pe[pe];
    }

    [[nodiscard]] inline MPI_Comm communicator() const {
        return _communicator;
    }

    // Functions to steal members of this graph

    auto&& take_node_distribution() {
        return std::move(_node_distribution);
    }
    auto&& take_edge_distribution() {
        return std::move(_edge_distribution);
    }
    auto&& take_nodes() {
        return std::move(_nodes);
    }
    auto&& take_edges() {
        return std::move(_edges);
    }
    auto&& take_node_weights() {
        return std::move(_node_weights);
    }
    auto&& take_edge_weights() {
        return std::move(_edge_weights);
    }
    auto&& take_ghost_owner() {
        return std::move(_ghost_owner);
    }
    auto&& take_ghost_to_global() {
        return std::move(_ghost_to_global);
    }
    auto&& take_global_to_ghost() {
        return std::move(_global_to_ghost);
    }

    // Debug functions

    void print() const;

private:
    void init_degree_buckets();
    void init_total_node_weight();
    void init_communication_metrics();

    NodeID       _n;
    EdgeID       _m;
    NodeID       _ghost_n;
    GlobalNodeID _offset_n;
    GlobalEdgeID _offset_m;
    GlobalNodeID _global_n;
    GlobalEdgeID _global_m;

    NodeWeight       _total_node_weight{};
    GlobalNodeWeight _global_total_node_weight{};
    NodeWeight       _max_node_weight{};
    GlobalNodeWeight _global_max_node_weight{};

    scalable_vector<GlobalNodeID> _node_distribution{};
    scalable_vector<GlobalEdgeID> _edge_distribution{};

    scalable_vector<EdgeID>     _nodes{};
    scalable_vector<NodeID>     _edges{};
    scalable_vector<NodeWeight> _node_weights{};
    scalable_vector<EdgeWeight> _edge_weights{};

    scalable_vector<PEID>         _ghost_owner{};
    scalable_vector<GlobalNodeID> _ghost_to_global{};
    growt::StaticGhostNodeMapping _global_to_ghost{};

    std::vector<EdgeID> _edge_cut_to_pe{};
    std::vector<EdgeID> _comm_vol_to_pe{};

    bool                _sorted;
    std::vector<NodeID> _buckets           = std::vector<NodeID>(shm::kNumberOfDegreeBuckets + 1);
    std::size_t         _number_of_buckets = 0;

    MPI_Comm _communicator;
};

class DistributedPartitionedGraph {
public:
    using NodeID           = DistributedGraph::NodeID;
    using GlobalNodeID     = DistributedGraph::GlobalNodeID;
    using NodeWeight       = DistributedGraph::NodeWeight;
    using GlobalNodeWeight = DistributedGraph::GlobalNodeWeight;
    using EdgeID           = DistributedGraph::EdgeID;
    using GlobalEdgeID     = DistributedGraph::GlobalEdgeID;
    using EdgeWeight       = DistributedGraph::EdgeWeight;
    using GlobalEdgeWeight = DistributedGraph::GlobalEdgeWeight;
    using BlockID          = ::dkaminpar::BlockID;
    using BlockWeight      = ::dkaminpar::BlockWeight;

    using block_weights_vector = scalable_vector<shm::parallel::Atomic<BlockWeight>>;

    DistributedPartitionedGraph(
        const DistributedGraph* graph, const BlockID k, scalable_vector<Atomic<BlockID>> partition,
        block_weights_vector block_weights)
        : _graph{graph},
          _k{k},
          _partition{std::move(partition)},
          _block_weights{std::move(block_weights)} {
        KASSERT(_partition.size() == _graph->total_n());
        KASSERT([&] {
            for (const BlockID b: _partition) {
                KASSERT(b < _k);
            }
        }());
    }

    DistributedPartitionedGraph(const DistributedGraph* graph, const BlockID k)
        : DistributedPartitionedGraph(
            graph, k, scalable_vector<Atomic<BlockID>>(graph->total_n()), block_weights_vector(graph->total_n())) {}

    DistributedPartitionedGraph() : _graph{nullptr}, _k{0}, _partition{} {}

    DistributedPartitionedGraph(const DistributedPartitionedGraph&) = delete;
    DistributedPartitionedGraph& operator=(const DistributedPartitionedGraph&) = delete;
    DistributedPartitionedGraph(DistributedPartitionedGraph&&) noexcept        = default;
    DistributedPartitionedGraph& operator=(DistributedPartitionedGraph&&) noexcept = default;

    [[nodiscard]] const DistributedGraph& graph() const {
        return *_graph;
    }
    void UNSAFE_set_graph(const DistributedGraph* graph) {
        _graph = graph;
    }

    // Delegates to _graph
    // clang-format off
  [[nodiscard]] inline GlobalNodeID global_n() const { return _graph->global_n(); }
  [[nodiscard]] inline GlobalEdgeID global_m() const { return _graph->global_m(); }
  [[nodiscard]] inline NodeID n() const { return _graph->n(); }
  [[nodiscard]] inline NodeID ghost_n() const { return _graph->ghost_n(); }
  [[nodiscard]] inline NodeID total_n() const { return _graph->total_n(); }
  [[nodiscard]] inline EdgeID m() const { return _graph->m(); }
  [[nodiscard]] inline GlobalNodeID offset_n(const PEID pe) const { return _graph->offset_n(pe); }
  [[nodiscard]] inline GlobalNodeID offset_n() const { return _graph->offset_n(); }
  [[nodiscard]] inline GlobalEdgeID offset_m() const { return _graph->offset_m(); }
  [[nodiscard]] inline NodeWeight total_node_weight() const { return _graph->total_node_weight(); }
  [[nodiscard]] inline NodeWeight max_node_weight() const { return _graph->max_node_weight(); }
  [[nodiscard]] inline bool contains_global_node(const GlobalNodeID global_u) const { return _graph->contains_global_node(global_u); }
  [[nodiscard]] inline bool contains_local_node(const NodeID local_u) const { return _graph->contains_local_node(local_u); }
  [[nodiscard]] inline bool is_ghost_node(const NodeID u) const { return _graph->is_ghost_node(u); }
  [[nodiscard]] inline bool is_owned_node(const NodeID u) const { return _graph->is_owned_node(u); }
  [[nodiscard]] inline PEID find_owner_of_global_node(const GlobalNodeID u) const { return _graph->find_owner_of_global_node(u); }
  [[nodiscard]] inline PEID ghost_owner(const NodeID u) const { return _graph->ghost_owner(u); }
  [[nodiscard]] inline GlobalNodeID local_to_global_node(const NodeID local_u) const { return _graph->local_to_global_node(local_u); }
  [[nodiscard]] inline NodeID global_to_local_node(const GlobalNodeID global_u) const { return _graph->global_to_local_node(global_u); }
  [[nodiscard]] inline NodeWeight node_weight(const NodeID u) const { return _graph->node_weight(u); }
  [[nodiscard]] inline EdgeWeight edge_weight(const EdgeID e) const { return _graph->edge_weight(e); }
  [[nodiscard]] inline EdgeID first_edge(const NodeID u) const { return _graph->first_edge(u); }
  [[nodiscard]] inline EdgeID first_invalid_edge(const NodeID u) const { return _graph->first_invalid_edge(u); }
  [[nodiscard]] inline NodeID edge_target(const EdgeID e) const { return _graph->edge_target(e); }
  [[nodiscard]] inline NodeID degree(const NodeID u) const { return _graph->degree(u); }
  [[nodiscard]] inline const auto &node_distribution() const { return _graph->node_distribution(); }
  [[nodiscard]] inline GlobalNodeID node_distribution(const PEID pe) const { return _graph->node_distribution(pe); }
  [[nodiscard]] inline const auto &edge_distribution() const { return _graph->edge_distribution(); }
  [[nodiscard]] inline GlobalEdgeID edge_distribution(const PEID pe) const { return _graph->edge_distribution(pe); }
  [[nodiscard]] const auto &raw_nodes() const { return _graph->raw_nodes(); }
  [[nodiscard]] const auto &raw_node_weights() const { return _graph->raw_node_weights(); }
  [[nodiscard]] const auto &raw_edges() const { return _graph->raw_edges(); }
  [[nodiscard]] const auto &raw_edge_weights() const { return _graph->raw_edge_weights(); }
  template<typename Lambda> inline void pfor_nodes(const NodeID from, const NodeID to, Lambda &&l) const { _graph->pfor_nodes(from, to, std::forward<Lambda>(l)); }
  template<typename Lambda> inline void pfor_nodes_range(const NodeID from, const NodeID to, Lambda &&l) const { _graph->pfor_nodes_range(from, to, std::forward<Lambda>(l)); }
  template<typename Lambda> inline void pfor_nodes(Lambda &&l) const { _graph->pfor_nodes(std::forward<Lambda>(l)); }
  template<typename Lambda> inline void pfor_nodes_range(Lambda &&l) const { _graph->pfor_nodes_range(std::forward<Lambda>(l)); }
  template<typename Lambda> inline void pfor_edges(Lambda &&l) const { _graph->pfor_edges(std::forward<Lambda>(l)); }
  [[nodiscard]] inline auto nodes() const { return _graph->nodes(); }
  [[nodiscard]] inline auto ghost_nodes() const { return _graph->ghost_nodes(); }
  [[nodiscard]] inline auto all_nodes() const { return _graph->all_nodes(); }
  [[nodiscard]] inline auto edges() const { return _graph->edges(); }
  [[nodiscard]] inline auto incident_edges(const NodeID u) const { return _graph->incident_edges(u); }
  [[nodiscard]] inline auto adjacent_nodes(const NodeID u) const { return _graph->adjacent_nodes(u); }
  [[nodiscard]] inline auto neighbors(const NodeID u) const { return _graph->neighbors(u); }
  [[nodiscard]] inline std::size_t bucket_size(const std::size_t bucket) const { return _graph->bucket_size(bucket); }
  [[nodiscard]] inline NodeID first_node_in_bucket(const std::size_t bucket) const { return _graph->first_node_in_bucket(bucket); }
  [[nodiscard]] inline NodeID first_invalid_node_in_bucket(const std::size_t bucket) const { return _graph->first_invalid_node_in_bucket(bucket); }
  [[nodiscard]] inline std::size_t number_of_buckets() const { return _graph->number_of_buckets(); }
  [[nodiscard]] inline bool sorted() const { return _graph->sorted(); }
  [[nodiscard]] inline EdgeID edge_cut_to_pe(const PEID pe) const { return _graph->edge_cut_to_pe(pe); }
  [[nodiscard]] inline EdgeID comm_vol_to_pe(const PEID pe) const { return _graph->comm_vol_to_pe(pe); }
  [[nodiscard]] MPI_Comm communicator() const { return _graph->communicator(); }
    // clang-format on

    [[nodiscard]] BlockID k() const {
        return _k;
    }

    template <typename Lambda>
    inline void pfor_blocks(Lambda&& l) const {
        tbb::parallel_for(static_cast<BlockID>(0), k(), std::forward<Lambda&&>(l));
    }

    [[nodiscard]] inline auto blocks() const {
        return shm::IotaRange<BlockID>(0, k());
    }

    [[nodiscard]] BlockID block(const NodeID u) const {
        ASSERT(u < _partition.size());
        return _partition[u].load(std::memory_order_relaxed);
    }

    template <bool update_block_weights = true>
    void set_block(const NodeID u, const BlockID b) {
        ASSERT(u < _graph->total_n());

        if constexpr (update_block_weights) {
            const NodeWeight u_weight = _graph->node_weight(u);
            _block_weights[_partition[u]] -= u_weight;
            _block_weights[b] += u_weight;
        }
        _partition[u].store(b, std::memory_order_relaxed);
    }

    [[nodiscard]] inline BlockWeight block_weight(const BlockID b) const {
        ASSERT(b < k());
        ASSERT(b < _block_weights.size());
        return _block_weights[b].load(std::memory_order_relaxed);
    }

    void set_block_weight(const BlockID b, const BlockWeight weight) {
        KASSERT(b < k());
        KASSERT(b < _block_weights.size());
        _block_weights[b].store(weight, std::memory_order_relaxed);
    }

    [[nodiscard]] const auto& block_weights() const {
        return _block_weights;
    }

    [[nodiscard]] scalable_vector<BlockWeight> block_weights_copy() const {
        scalable_vector<BlockWeight> copy(k());
        pfor_blocks([&](const BlockID b) { copy[b] = block_weight(b); });
        return copy;
    }

    [[nodiscard]] auto&& take_block_weights() {
        return std::move(_block_weights);
    }

    [[nodiscard]] const auto& partition() const {
        return _partition;
    }
    [[nodiscard]] auto&& take_partition() {
        return std::move(_partition);
    }

    [[nodiscard]] scalable_vector<BlockID> copy_partition() const {
        scalable_vector<BlockID> copy(n());
        pfor_nodes([&](const NodeID u) { copy[u] = block(u); });
        return copy;
    }

private:
    const DistributedGraph*              _graph;
    BlockID                              _k;
    scalable_vector<Atomic<BlockID>>     _partition;
    scalable_vector<Atomic<BlockWeight>> _block_weights;
};

namespace graph {
/**
 * Prints verbose statistics on the distribution of the graph across PEs and the number of ghost nodes, but only if
 * verbose statistics are enabled as build option.
 * @param graph Graph for which statistics are printed.
 */
void print_verbose_stats(const DistributedGraph& graph);
} // namespace graph

namespace graph::debug {
// validate structure of a distributed graph
bool validate(const DistributedGraph& graph, int root = 0);

// validate structure of a distributed graph partition
bool validate_partition(const DistributedPartitionedGraph& p_graph);
} // namespace graph::debug
} // namespace dkaminpar
