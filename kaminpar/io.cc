/*******************************************************************************
 * @file:   io.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:  Graph and partition IO functions.
 ******************************************************************************/
#include "kaminpar/io.h"

#include "common/timer.h"

namespace kaminpar::shm::io {
namespace metis {
void write_file(
    std::ofstream& out, const StaticArray<EdgeID>& nodes, const StaticArray<NodeID>& edges,
    const StaticArray<NodeWeight>& node_weights, const StaticArray<EdgeWeight>& edge_weights, const std::string& comment
) {
    const bool write_node_weights = !node_weights.empty();
    const bool write_edge_weights = !edge_weights.empty();

    if (!comment.empty()) {
        out << "% " << comment << "\n";
    }

    // header
    out << nodes.size() - 1 << " " << edges.size() / 2;
    if (write_node_weights || write_edge_weights) {
        out << " " << static_cast<int>(write_node_weights) << static_cast<int>(write_edge_weights);
    }
    out << "\n";

    // content
    for (NodeID u = 0; u < nodes.size() - 1; ++u) {
        if (write_node_weights) {
            out << node_weights[u] << " ";
        }
        for (EdgeID e = nodes[u]; e < nodes[u + 1]; ++e) {
            out << edges[e] + 1 << " ";
            if (write_edge_weights) {
                out << edge_weights[e] << " ";
            }
        }
        out << "\n";
    }
}

void write_file(
    const std::string& filename, const StaticArray<EdgeID>& nodes, const StaticArray<NodeID>& edges,
    const StaticArray<NodeWeight>& node_weights, const StaticArray<EdgeWeight>& edge_weights, const std::string& comment
) {
    std::ofstream out(filename);
    if (!out) {
        FATAL_PERROR << "Error while opening " << filename;
    }
    write_file(out, nodes, edges, node_weights, edge_weights, comment);
}
} // namespace metis

//
// Public Metis functions
//

namespace metis {
Statistics read(
    const std::string& filename, StaticArray<EdgeID>& nodes, StaticArray<NodeID>& edges,
    StaticArray<NodeWeight>& node_weights, StaticArray<EdgeWeight>& edge_weights
) {
    using namespace kaminpar::io::metis;

    bool store_node_weights = false;
    bool store_edge_weights = false;

    NodeID u = 0;
    EdgeID e = 0;

    Statistics stats;

    parse(
        filename,
        [&](const auto& format) {
            store_node_weights = format.has_node_weights;
            store_edge_weights = format.has_edge_weights;
            nodes.resize(format.number_of_nodes + 1);
            edges.resize(format.number_of_edges * 2);
            if (store_node_weights) {
                node_weights.resize(format.number_of_nodes);
            }
            if (store_edge_weights) {
                edge_weights.resize(format.number_of_edges * 2);
            }
        },
        [&](const std::uint64_t& weight) {
            stats.total_node_weight += weight;
            stats.has_isolated_nodes |= (u > 0 && nodes[u - 1] == e);

            if (store_node_weights) {
                node_weights[u] = static_cast<NodeWeight>(weight);
            }
            nodes[u] = e;
            ++u;
        },
        [&](const std::uint64_t& weight, const std::uint64_t& v) {
            stats.total_edge_weight += weight;
            if (store_edge_weights) {
                edge_weights[e] = static_cast<EdgeWeight>(weight);
            }
            edges[e] = static_cast<NodeID>(v);
            ++e;
        }
    );
    nodes[u] = e;

    // only keep weights if the graph is really weighted
    const bool unit_node_weights = stats.total_node_weight + 1 == nodes.size();
    const bool unit_edge_weights = stats.total_edge_weight == edges.size();
    if (unit_node_weights) {
        node_weights.free();
    }
    if (unit_edge_weights) {
        edge_weights.free();
    }

    return stats;
}

Graph read(const std::string& filename, bool ignore_node_weights, bool ignore_edge_weights) {
    StaticArray<EdgeID>     nodes;
    StaticArray<NodeID>     edges;
    StaticArray<NodeWeight> node_weights;
    StaticArray<EdgeWeight> edge_weights;
    metis::read(filename, nodes, edges, node_weights, edge_weights);

    if (ignore_node_weights) {
        node_weights.free();
    }
    if (ignore_edge_weights) {
        edge_weights.free();
    }

    return {std::move(nodes), std::move(edges), std::move(node_weights), std::move(edge_weights)};
}

void write(const std::string& filename, const Graph& graph, const std::string& comment) {
    metis::write_file(
        filename, graph.raw_nodes(), graph.raw_edges(), graph.raw_node_weights(), graph.raw_edge_weights(), comment
    );
}
} // namespace metis

//
// Partition
//

namespace partition {
void write(const std::string& filename, const StaticArray<BlockID>& partition) {
    std::ofstream out(filename);
    for (const BlockID block: partition) {
        out << block << "\n";
    }
}

void write(const std::string& filename, const PartitionedGraph& p_graph) {
    write(filename, p_graph.partition());
}

void write(const std::string& filename, const StaticArray<BlockID>& partition, const StaticArray<NodeID>& permutation) {
    std::ofstream out(filename);
    for (const NodeID u: permutation) {
        out << partition[u] << "\n";
    }
}

void write(const std::string& filename, const PartitionedGraph& p_graph, const StaticArray<NodeID>& permutation) {
    write(filename, p_graph.partition(), permutation);
}
} // namespace partition
} // namespace kaminpar::shm::io
