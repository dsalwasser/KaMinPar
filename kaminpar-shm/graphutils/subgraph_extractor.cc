/*******************************************************************************
 * @file:   subgraph_extraction.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:  Extracts the subgraphs induced by each block of a partition.
 ******************************************************************************/
#include "kaminpar-shm/graphutils/subgraph_extractor.h"

#include <array>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include "kaminpar-shm/datastructures/block_induced_subgraph.h"
#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/metrics.h"
#include "kaminpar-shm/partitioning/partition_utils.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/parallel/algorithm.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm::graph {
namespace {
SET_DEBUG(false);

template <typename Graph>
SequentialSubgraphExtractionResult extract_subgraphs_sequential_generic_graph(
    const PartitionedGraph &p_graph,
    const Graph &graph,
    const std::array<BlockID, 2> &final_ks,
    const SubgraphMemoryStartPosition memory_position,
    SubgraphMemory &subgraph_memory,
    TemporarySubgraphMemory &tmp_subgraph_memory
) {
  KASSERT(p_graph.k() == 2u, "Only suitable for bipartitions!", assert::light);

  const bool is_node_weighted = graph.is_node_weighted();
  const bool is_edge_weighted = graph.is_edge_weighted();

  const BlockID final_k = final_ks[0] + final_ks[1];
  tmp_subgraph_memory.ensure_size_nodes(graph.n() + final_k, is_node_weighted);

  auto &nodes = tmp_subgraph_memory.nodes;
  auto &edges = tmp_subgraph_memory.edges;
  auto &node_weights = tmp_subgraph_memory.node_weights;
  auto &edge_weights = tmp_subgraph_memory.edge_weights;
  auto &mapping = tmp_subgraph_memory.mapping;

  std::array<NodeID, 2> s_n{0, 0};
  std::array<EdgeID, 2> s_m{0, 0};

  // find graph sizes
  for (const NodeID u : graph.nodes()) {
    const BlockID b = p_graph.block(u);
    tmp_subgraph_memory.mapping[u] = s_n[b]++;

    graph.adjacent_nodes(u, [&](const NodeID v) {
      if (p_graph.block(v) == b) {
        ++s_m[b];
      }
    });
  }

  // start position of subgraph[1] in common memory ds
  const NodeID n1 = s_n[0] + final_ks[0];
  const EdgeID m1 = s_m[0];

  nodes[0] = 0;
  nodes[n1] = 0;
  tmp_subgraph_memory.ensure_size_edges(s_m[0] + s_m[1], is_edge_weighted);

  // build extract graphs in temporary memory buffer
  std::array<EdgeID, 2> next_edge_id{0, 0};

  for (const NodeID u : graph.nodes()) {
    const BlockID b = p_graph.block(u);

    const NodeID n0 = b * n1;
    const EdgeID m0 = b * m1; // either 0 or s_m[0]

    graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
      if (p_graph.block(v) == b) {
        edges[m0 + next_edge_id[b]] = mapping[v];
        if (is_edge_weighted) {
          edge_weights[m0 + next_edge_id[b]] = w;
        }
        ++next_edge_id[b];
      }
    });

    nodes[n0 + mapping[u] + 1] = next_edge_id[b];
    if (is_node_weighted) {
      node_weights[n0 + mapping[u]] = graph.node_weight(u);
    }
  }

  // copy graphs to subgraph_memory at memory_position
  // THIS OPERATION OVERWRITES p_graph!
  std::copy(
      nodes.begin(),
      nodes.begin() + graph.n() + final_k,
      subgraph_memory.nodes.begin() + memory_position.nodes_start_pos
  );
  std::copy(
      edges.begin(),
      edges.begin() + s_m[0] + s_m[1],
      subgraph_memory.edges.begin() + memory_position.edges_start_pos
  );
  if (is_node_weighted) {
    std::copy(
        node_weights.begin(),
        node_weights.begin() + graph.n() + final_k,
        subgraph_memory.node_weights.begin() + memory_position.nodes_start_pos
    );
  }
  if (is_edge_weighted) {
    std::copy(
        edge_weights.begin(),
        edge_weights.begin() + s_m[0] + s_m[1],
        subgraph_memory.edge_weights.begin() + memory_position.edges_start_pos
    );
  }

  std::array<SubgraphMemoryStartPosition, 2> subgraph_positions;
  subgraph_positions[0].nodes_start_pos = memory_position.nodes_start_pos;
  subgraph_positions[0].edges_start_pos = memory_position.edges_start_pos;
  subgraph_positions[1].nodes_start_pos = memory_position.nodes_start_pos + n1;
  subgraph_positions[1].edges_start_pos = memory_position.edges_start_pos + m1;

  auto create_graph = [&](const NodeID n0, const NodeID n, const EdgeID m0, const EdgeID m) {
    StaticArray<EdgeID> s_nodes(
        n + 1, subgraph_memory.nodes.data() + memory_position.nodes_start_pos + n0
    );
    StaticArray<NodeID> s_edges(
        m, subgraph_memory.edges.data() + memory_position.edges_start_pos + m0
    );
    StaticArray<NodeWeight> s_node_weights(
        is_node_weighted * n,
        subgraph_memory.node_weights.data() +
            is_node_weighted * (memory_position.nodes_start_pos + n0)
    );
    StaticArray<EdgeWeight> s_edge_weights(
        is_edge_weighted * m,
        subgraph_memory.edge_weights.data() +
            is_edge_weighted * (memory_position.edges_start_pos + m0)
    );
    return shm::Graph(std::make_unique<CSRGraph>(
        CSRGraph::seq{},
        std::move(s_nodes),
        std::move(s_edges),
        std::move(s_node_weights),
        std::move(s_edge_weights)
    ));
  };

  std::array<shm::Graph, 2> subgraphs{
      create_graph(0, s_n[0], 0, s_m[0]), create_graph(n1, s_n[1], m1, s_m[1])
  };

  return {std::move(subgraphs), std::move(subgraph_positions)};
}
} // namespace

SequentialSubgraphExtractionResult extract_subgraphs_sequential(
    const PartitionedGraph &p_graph,
    const std::array<BlockID, 2> &final_ks,
    const SubgraphMemoryStartPosition memory_position,
    SubgraphMemory &subgraph_memory,
    TemporarySubgraphMemory &tmp_subgraph_memory
) {
  return p_graph.reified([&](const auto &graph) {
    return extract_subgraphs_sequential_generic_graph(
        p_graph, graph, final_ks, memory_position, subgraph_memory, tmp_subgraph_memory
    );
  });
}

namespace {
template <typename Graph>
SequentialSpanSubgraphExtractionResult extract_span_subgraphs_sequential_generic_graph(
    const PartitionedGraph &p_graph,
    Graph &graph,
    const std::array<BlockID, 2> &final_ks,
    const BlockID b_test
) {
  KASSERT(p_graph.k() == 2u, "Only suitable for bipartitions!", assert::light);

  StaticArray<NodeID> &global_to_local = graph.global_to_local();
  LocalToGlobalMapping &local_to_global = graph.local_to_global();
  StaticArray<std::uint8_t> &border_nodes = graph.border_nodes();

  std::array<NodeID, 2> s_n{0, 0};
  std::array<EdgeID, 2> s_m{0, 0};

  std::array<NodeWeight, 2> s_max_node_weight{0, 0};
  std::array<NodeWeight, 2> s_total_node_weight{0, 0};
  std::array<EdgeWeight, 2> s_total_edge_weight{0, 0};

  for (const NodeID u : graph.nodes()) {
    const BlockID b = p_graph.block(u);
    s_n[b] += 1;

    const NodeWeight u_weight = graph.node_weight(u);
    s_max_node_weight[b] = std::max(s_max_node_weight[b], u_weight);
    s_total_node_weight[b] += u_weight;

    const NodeID u_global = local_to_global[u];
    std::uint8_t is_border_node = border_nodes[u_global];

    graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
      if (p_graph.block(v) != b) {
        is_border_node = 1;
        return;
      }

      s_m[b] += 1;
      s_total_edge_weight[b] += w;
    });

    border_nodes[u_global] = is_border_node;
  }

  const auto construct_local_to_global_mapping = [&](const BlockID b) {
    return LocalToGlobalMapping(
        !local_to_global.use_shared(),
        local_to_global.offset() + (b == 0 ? 0 : s_n[0]),
        local_to_global.shared(),
        local_to_global.backup()
    );
  };
  std::array<LocalToGlobalMapping, 2> s_local_to_global{
      construct_local_to_global_mapping(0), construct_local_to_global_mapping(1)
  };

  std::array<NodeID, 2> s_i{0, 0};
  for (const NodeID u : graph.nodes()) {
    const BlockID b = p_graph.block(u);
    const NodeID u_local = s_i[b]++;

    global_to_local[local_to_global[u]] = u_local;
    s_local_to_global[b][u_local] = local_to_global[u];
  }

  const auto create_graph = [&](const BlockID b) {
    return shm::Graph(std::make_unique<Graph>(
        graph.underlying_graph(),
        graph.partition(),
        s_n[b],
        s_m[b],
        s_max_node_weight[b],
        s_total_node_weight[b],
        s_total_edge_weight[b],
        b_test + b,
        StaticArray<NodeID>(global_to_local.size(), global_to_local.data()),
        std::move(s_local_to_global[b]),
        StaticArray<std::uint8_t>(border_nodes.size(), border_nodes.data())
    ));
  };
  std::array<shm::Graph, 2> subgraphs{create_graph(0), create_graph(1)};

  return {std::move(subgraphs)};
}
} // namespace

SequentialSpanSubgraphExtractionResult extract_span_subgraphs_sequential(
    PartitionedGraph &p_graph, const std::array<BlockID, 2> &final_ks, const BlockID b_test
) {
  return p_graph.block_induced_reified([&](auto &graph) {
    return extract_span_subgraphs_sequential_generic_graph(p_graph, graph, final_ks, b_test);
  });
}

namespace {
template <typename Graph>
SequentialSubgraphExtractionResult extract_subgraphs_sequential_generic_graph(
    const PartitionedGraph &p_graph,
    const Graph &graph,
    const std::array<BlockID, 2> &final_ks,
    const SubgraphMemoryStartPosition memory_position,
    OCSubgraphMemory &subgraph_memory,
    OCTemporarySubgraphMemory &tmp_subgraph_memory
) {
  KASSERT(p_graph.k() == 2u, "Only suitable for bipartitions!", assert::light);

  const bool is_node_weighted = graph.is_node_weighted();
  const bool is_edge_weighted = graph.is_edge_weighted();

  const BlockID final_k = final_ks[0] + final_ks[1];

  NodeID *mapping = tmp_subgraph_memory.mapping();
  EdgeID *nodes = tmp_subgraph_memory.nodes();
  NodeID *edges = tmp_subgraph_memory.edges();
  NodeWeight *node_weights = tmp_subgraph_memory.node_weights();
  EdgeWeight *edge_weights = tmp_subgraph_memory.edge_weights();

  std::array<NodeID, 2> s_n{0, 0};
  std::array<EdgeID, 2> s_m{0, 0};

  // find graph sizes
  for (const NodeID u : graph.nodes()) {
    const BlockID b = p_graph.block(u);
    mapping[u] = s_n[b]++;

    graph.adjacent_nodes(u, [&](const NodeID v) {
      if (p_graph.block(v) == b) {
        ++s_m[b];
      }
    });
  }

  // start position of subgraph[1] in common memory ds
  const NodeID n1 = s_n[0] + final_ks[0];
  const EdgeID m1 = s_m[0];

  nodes[0] = 0;
  nodes[n1] = 0;

  // build extract graphs in temporary memory buffer
  std::array<EdgeID, 2> next_edge_id{0, 0};

  for (const NodeID u : graph.nodes()) {
    const BlockID b = p_graph.block(u);

    const NodeID n0 = b * n1;
    const EdgeID m0 = b * m1; // either 0 or s_m[0]

    graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
      if (p_graph.block(v) == b) {
        edges[m0 + next_edge_id[b]] = mapping[v];
        if (is_edge_weighted) {
          edge_weights[m0 + next_edge_id[b]] = w;
        }
        ++next_edge_id[b];
      }
    });

    nodes[n0 + mapping[u] + 1] = next_edge_id[b];
    if (is_node_weighted) {
      node_weights[n0 + mapping[u]] = graph.node_weight(u);
    }
  }

  // copy graphs to subgraph_memory at memory_position
  // THIS OPERATION OVERWRITES p_graph!
  std::copy(
      nodes, nodes + graph.n() + final_k, subgraph_memory.nodes() + memory_position.nodes_start_pos
  );
  std::copy(
      edges, edges + s_m[0] + s_m[1], subgraph_memory.edges() + memory_position.edges_start_pos
  );
  if (is_node_weighted) {
    std::copy(
        node_weights,
        node_weights + graph.n() + final_k,
        subgraph_memory.node_weights() + memory_position.nodes_start_pos
    );
  }
  if (is_edge_weighted) {
    std::copy(
        edge_weights,
        edge_weights + s_m[0] + s_m[1],
        subgraph_memory.edge_weights() + memory_position.edges_start_pos
    );
  }
  tmp_subgraph_memory.record(
      graph.n() + final_k, s_m[0] + s_m[1], is_node_weighted, is_edge_weighted
  );

  std::array<SubgraphMemoryStartPosition, 2> subgraph_positions;
  subgraph_positions[0].nodes_start_pos = memory_position.nodes_start_pos;
  subgraph_positions[0].edges_start_pos = memory_position.edges_start_pos;
  subgraph_positions[1].nodes_start_pos = memory_position.nodes_start_pos + n1;
  subgraph_positions[1].edges_start_pos = memory_position.edges_start_pos + m1;

  auto create_graph = [&](const NodeID n0, const NodeID n, const EdgeID m0, const EdgeID m) {
    StaticArray<EdgeID> s_nodes(
        n + 1, subgraph_memory.nodes() + memory_position.nodes_start_pos + n0
    );
    StaticArray<NodeID> s_edges(m, subgraph_memory.edges() + memory_position.edges_start_pos + m0);
    StaticArray<NodeWeight> s_node_weights(
        is_node_weighted * n,
        subgraph_memory.node_weights() + is_node_weighted * (memory_position.nodes_start_pos + n0)
    );
    StaticArray<EdgeWeight> s_edge_weights(
        is_edge_weighted * m,
        subgraph_memory.edge_weights() + is_edge_weighted * (memory_position.edges_start_pos + m0)
    );
    return shm::Graph(std::make_unique<CSRGraph>(
        CSRGraph::seq{},
        std::move(s_nodes),
        std::move(s_edges),
        std::move(s_node_weights),
        std::move(s_edge_weights)
    ));
  };

  std::array<shm::Graph, 2> subgraphs{
      create_graph(0, s_n[0], 0, s_m[0]), create_graph(n1, s_n[1], m1, s_m[1])
  };

  return {std::move(subgraphs), std::move(subgraph_positions)};
}
} // namespace

SequentialSubgraphExtractionResult extract_subgraphs_sequential(
    const PartitionedGraph &p_graph,
    const std::array<BlockID, 2> &final_ks,
    const SubgraphMemoryStartPosition memory_position,
    OCSubgraphMemory &subgraph_memory,
    OCTemporarySubgraphMemory &tmp_subgraph_memory
) {
  return p_graph.reified([&](const auto &graph) {
    return extract_subgraphs_sequential_generic_graph(
        p_graph, graph, final_ks, memory_position, subgraph_memory, tmp_subgraph_memory
    );
  });
}

OCSubgraphMemoryPreprocessingResult extract_subgraphs_preprocessing(const PartitionedGraph &p_graph
) {
  const BlockID n = p_graph.n();
  const BlockID k = p_graph.k();

  StaticArray<NodeID> mapping(n, static_array::noinit);
  StaticArray<NodeID> block_nodes_offset(p_graph.k() + 1);
  StaticArray<NodeID> block_nodes(n, static_array::noinit);

  tbb::enumerable_thread_specific<ScalableVector<NodeID>> tl_num_nodes_in_block{[&] {
    return ScalableVector<NodeID>(k);
  }};
  tbb::parallel_for(tbb::blocked_range<NodeID>(0, n), [&](auto &local_nodes) {
    auto &num_nodes_in_block = tl_num_nodes_in_block.local();

    const NodeID first_node = local_nodes.begin();
    const NodeID last_node = local_nodes.end();

    for (NodeID u = first_node; u < last_node; ++u) {
      const BlockID b = p_graph.block(u);
      num_nodes_in_block[b] += 1;
    }
  });

  tbb::parallel_for<BlockID>(0, k, [&](const BlockID b) {
    NodeID num_block_nodes = 0;
    for (const auto &local_num_nodes : tl_num_nodes_in_block) {
      num_block_nodes += local_num_nodes[b];
    }
    block_nodes_offset[b + 1] = num_block_nodes;
  });
  parallel::prefix_sum(
      block_nodes_offset.begin(), block_nodes_offset.end(), block_nodes_offset.begin()
  );

  StaticArray<NodeID> block_node_index(p_graph.k());
  tbb::parallel_for(tbb::blocked_range<NodeID>(0, n), [&](auto &local_nodes) {
    const NodeID first_node = local_nodes.begin();
    const NodeID last_node = local_nodes.end();

    for (NodeID u = first_node; u < last_node; ++u) {
      const BlockID b = p_graph.block(u);
      const NodeID offset = block_nodes_offset[b];
      const NodeID i = __atomic_fetch_add(&block_node_index[b], 1, __ATOMIC_RELAXED);
      block_nodes[offset + i] = u;
      mapping[u] = i;
    }
  });

  return {std::move(mapping), std::move(block_nodes_offset), std::move(block_nodes)};
}

namespace {
template <typename Graph>
shm::Graph extract_subgraph_generic_graph(
    const PartitionedGraph &p_graph,
    const Graph &graph,
    const BlockID block,
    const StaticArray<NodeID> &block_nodes,
    const StaticArray<NodeID> &mapping,
    graph::OCSubgraphMemory &subgraph_memory
) {
  const bool has_node_weights = graph.is_node_weighted();
  const bool has_edge_weights = graph.is_edge_weighted();

  EdgeID *nodes = subgraph_memory.nodes();
  NodeID *edges = subgraph_memory.edges();
  NodeWeight *node_weights = subgraph_memory.node_weights();
  EdgeWeight *edge_weights = subgraph_memory.edge_weights();

  NodeID cur_node = 0;
  EdgeID cur_edge = 0;
  for (const NodeID u : block_nodes) {
    nodes[cur_node] = cur_edge;
    if (has_node_weights) {
      node_weights[cur_node] = graph.node_weight(u);
    }
    cur_node += 1;

    graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
      if (p_graph.block(v) != block) {
        return;
      }

      edges[cur_edge] = mapping[v];
      if (has_edge_weights) {
        edge_weights[cur_edge] = w;
      }
      cur_edge += 1;
    });
  }
  nodes[block_nodes.size()] = cur_edge;

  subgraph_memory.record(cur_node, cur_edge, has_node_weights, has_edge_weights);
  return shm::Graph(std::make_unique<CSRGraph>(
      CSRGraph::seq{},
      StaticArray<EdgeID>(cur_node + 1, nodes),
      StaticArray<NodeID>(cur_edge, edges),
      has_node_weights ? StaticArray<NodeWeight>(cur_node, node_weights)
                       : StaticArray<NodeWeight>(),
      has_edge_weights ? StaticArray<EdgeWeight>(cur_edge, edge_weights) : StaticArray<EdgeWeight>()
  ));
}
} // namespace

Graph extract_subgraph(
    const PartitionedGraph &p_graph,
    const BlockID block,
    const StaticArray<NodeID> &block_nodes,
    const StaticArray<NodeID> &mapping,
    graph::OCSubgraphMemory &subgraph_memory
) {
  return p_graph.reified([&](const auto &concrete_graph) {
    return extract_subgraph_generic_graph(
        p_graph, concrete_graph, block, block_nodes, mapping, subgraph_memory
    );
  });
}

namespace {
/*
 * Builds a block-induced subgraph for each block of a partitioned graph. Return
 * type contains a mapping that maps nodes from p_graph to nodes in the
 * respective subgraph; we need this because the order in which nodes in
 * subgraphs appear is non-deterministic due to parallelization.
 */
template <typename Graph>
SubgraphExtractionResult extract_subgraphs_generic_graph(
    const PartitionedGraph &p_graph,
    const Graph &graph,
    const BlockID input_k,
    SubgraphMemory &subgraph_memory
) {
  START_TIMER("Allocation");
  StaticArray<NodeID> mapping(p_graph.n());
  StaticArray<SubgraphMemoryStartPosition> start_positions(p_graph.k() + 1);
  StaticArray<NodeID> bucket_index(p_graph.k());
  ScalableVector<shm::Graph> subgraphs(p_graph.k());
  STOP_TIMER();

  // count number of nodes and edges in each block
  START_TIMER("Count block size");
  tbb::enumerable_thread_specific<ScalableVector<NodeID>> tl_num_nodes_in_block{[&] {
    return ScalableVector<NodeID>(p_graph.k());
  }};
  tbb::enumerable_thread_specific<ScalableVector<EdgeID>> tl_num_edges_in_block{[&] {
    return ScalableVector<EdgeID>(p_graph.k());
  }};

  tbb::parallel_for(tbb::blocked_range<NodeID>(0, graph.n()), [&](auto &r) {
    auto &num_nodes_in_block = tl_num_nodes_in_block.local();
    auto &num_edges_in_block = tl_num_edges_in_block.local();

    for (NodeID u = r.begin(); u != r.end(); ++u) {
      const BlockID u_block = p_graph.block(u);
      ++num_nodes_in_block[u_block];
      graph.adjacent_nodes(u, [&](const NodeID v) {
        if (p_graph.block(v) == u_block) {
          ++num_edges_in_block[u_block];
        }
      });
    }
  });
  STOP_TIMER();

  START_TIMER("Merge block sizes");
  tbb::parallel_for<BlockID>(0, p_graph.k(), [&](const BlockID b) {
    NodeID num_nodes = partitioning::compute_final_k(
        b, p_graph.k(), input_k
    ); // padding for sequential subgraph extraction
    EdgeID num_edges = 0;
    for (auto &local_num_nodes : tl_num_nodes_in_block) {
      num_nodes += local_num_nodes[b];
    }
    for (auto &local_num_edges : tl_num_edges_in_block) {
      num_edges += local_num_edges[b];
    }
    start_positions[b + 1].nodes_start_pos = num_nodes;
    start_positions[b + 1].edges_start_pos = num_edges;
  });
  parallel::prefix_sum(start_positions.begin(), start_positions.end(), start_positions.begin());
  STOP_TIMER();

  // build temporary bucket array in nodes array
  START_TIMER("Build bucket array");
  tbb::parallel_for<NodeID>(0, p_graph.n(), [&](const NodeID u) {
    const BlockID b = p_graph.block(u);
    const NodeID pos_in_subgraph = __atomic_fetch_add(&bucket_index[b], 1, __ATOMIC_RELAXED);
    const NodeID pos = start_positions[b].nodes_start_pos + pos_in_subgraph;
    subgraph_memory.nodes[pos] = u;
    mapping[u] = pos_in_subgraph; // concurrent random access write
  });
  STOP_TIMER();

  const bool is_node_weighted = graph.is_node_weighted();
  const bool is_edge_weighted = graph.is_edge_weighted();

  // build graph
  START_TIMER("Construct subgraphs");
  tbb::parallel_for<BlockID>(0, p_graph.k(), [&](const BlockID b) {
    const NodeID nodes_start_pos = start_positions[b].nodes_start_pos;
    EdgeID e = 0;                                  // edge = in subgraph
    for (NodeID u = 0; u < bucket_index[b]; ++u) { // u = in subgraph
      const NodeID pos = nodes_start_pos + u;
      const NodeID u_prime = subgraph_memory.nodes[pos]; // u_prime = in graph
      subgraph_memory.nodes[pos] = e;
      if (is_node_weighted) {
        subgraph_memory.node_weights[pos] = graph.node_weight(u_prime);
      }

      const EdgeID e0 = start_positions[b].edges_start_pos;

      graph.adjacent_nodes(
          u_prime,
          [&](const NodeID v_prime, const EdgeWeight w_prime) { // v_prime, w_prime = in graph
            if (p_graph.block(v_prime) == b) {                  // only keep internal edges
              if (is_edge_weighted) {
                subgraph_memory.edge_weights[e0 + e] = w_prime;
              }
              subgraph_memory.edges[e0 + e] = mapping[v_prime];
              ++e;
            }
          }
      );
    }

    subgraph_memory.nodes[nodes_start_pos + bucket_index[b]] = e;
  });
  STOP_TIMER();

  START_TIMER("Create graph objects");
  tbb::parallel_for<BlockID>(0, p_graph.k(), [&](const BlockID b) {
    const NodeID n0 = start_positions[b].nodes_start_pos;
    const EdgeID m0 = start_positions[b].edges_start_pos;

    const NodeID n = start_positions[b + 1].nodes_start_pos - n0 -
                     partitioning::compute_final_k(b, p_graph.k(), input_k);
    const EdgeID m = start_positions[b + 1].edges_start_pos - m0;

    StaticArray<EdgeID> nodes(n + 1, subgraph_memory.nodes.data() + n0);
    StaticArray<NodeID> edges(m, subgraph_memory.edges.data() + m0);
    StaticArray<NodeWeight> node_weights(
        is_node_weighted * n, subgraph_memory.node_weights.data() + is_node_weighted * n0
    );
    StaticArray<EdgeWeight> edge_weights(
        is_edge_weighted * m, subgraph_memory.edge_weights.data() + is_edge_weighted * m0
    );
    subgraphs[b] = shm::Graph(std::make_unique<CSRGraph>(
        std::move(nodes), std::move(edges), std::move(node_weights), std::move(edge_weights)
    ));
  });
  STOP_TIMER();

  KASSERT(
      [&] {
        for (const BlockID b : p_graph.blocks()) {
          if (auto *csr_graph = dynamic_cast<CSRGraph *>(subgraphs[b].underlying_graph());
              csr_graph != nullptr) {
            if (!debug::validate_graph(*csr_graph)) {
              return false;
            }
          }
        }
        return true;
      }(),
      "",
      assert::heavy
  );

  return {std::move(subgraphs), std::move(mapping), std::move(start_positions)};
}
} // namespace

SubgraphExtractionResult extract_subgraphs(
    const PartitionedGraph &p_graph, const BlockID input_k, SubgraphMemory &subgraph_memory
) {
  return p_graph.reified([&](const auto &concrete_graph) {
    return extract_subgraphs_generic_graph(p_graph, concrete_graph, input_k, subgraph_memory);
  });
}

namespace {
template <typename Graph>
SpanSubgraphExtractionResult extract_span_subgraphs_generic_graph(
    PartitionedGraph &p_graph, const Graph &graph, BlockID input_k, NodeID *backup_local_to_global
) {
  SCOPED_TIMER("Extract subgraphs");
  SCOPED_HEAP_PROFILER("Extract subgraphs");

  const NodeID num_nodes = p_graph.n();
  const NodeID num_blocks = p_graph.k();
  StaticArray<std::uint8_t> border_nodes(num_nodes, static_array::noinit);

  std::vector<NodeID> local_num_nodes(num_blocks);
  std::vector<EdgeID> local_num_edges(num_blocks);
  std::vector<NodeWeight> local_max_node_weight(num_blocks);
  std::vector<NodeWeight> local_total_node_weight(num_blocks);
  std::vector<EdgeWeight> local_total_edge_weight(num_blocks);
  TIMED_SCOPE("Compute subgraph stats") {
    tbb::enumerable_thread_specific<ScalableVector<NodeID>> local_num_nodes_ets{[&] {
      return ScalableVector<NodeID>(num_blocks);
    }};
    tbb::enumerable_thread_specific<ScalableVector<EdgeID>> local_num_edges_ets{[&] {
      return ScalableVector<EdgeID>(num_blocks);
    }};
    tbb::enumerable_thread_specific<ScalableVector<NodeWeight>> local_max_node_weight_ets{[&] {
      return ScalableVector<NodeWeight>(num_blocks);
    }};
    tbb::enumerable_thread_specific<ScalableVector<NodeWeight>> local_total_node_weight_ets{[&] {
      return ScalableVector<NodeWeight>(num_blocks);
    }};
    tbb::enumerable_thread_specific<ScalableVector<EdgeWeight>> local_total_edge_weight_ets{[&] {
      return ScalableVector<EdgeWeight>(num_blocks);
    }};

    tbb::parallel_for(tbb::blocked_range<NodeID>(0, num_nodes), [&](auto &local_nodes) {
      auto &local_num_nodes = local_num_nodes_ets.local();
      auto &local_num_edges = local_num_edges_ets.local();
      auto &local_max_node_weight = local_max_node_weight_ets.local();
      auto &local_total_node_weight = local_total_node_weight_ets.local();
      auto &local_total_edge_weight = local_total_edge_weight_ets.local();

      const NodeID first_local_node = local_nodes.begin();
      const NodeID last_local_node = local_nodes.end();
      for (NodeID u = first_local_node; u < last_local_node; ++u) {
        const BlockID u_block = p_graph.block(u);
        const NodeWeight u_weight = graph.node_weight(u);

        local_num_nodes[u_block] += 1;
        local_total_node_weight[u_block] += u_weight;
        local_max_node_weight[u_block] = std::max(local_max_node_weight[u_block], u_weight);

        std::uint8_t is_border_node = 0;
        graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
          if (p_graph.block(v) != u_block) {
            is_border_node = 1;
            return;
          }

          local_num_edges[u_block] += 1;
          local_total_edge_weight[u_block] += w;
        });

        border_nodes[u] = is_border_node;
      }
    });

    tbb::parallel_for<BlockID>(0, num_blocks, [&](const BlockID b) {
      NodeID num_nodes = 0;
      for (const auto &local_num_nodes : local_num_nodes_ets) {
        num_nodes += local_num_nodes[b];
      }
      local_num_nodes[b] = num_nodes;

      EdgeID num_edges = 0;
      for (const auto &local_num_edges : local_num_edges_ets) {
        num_edges += local_num_edges[b];
      }
      local_num_edges[b] = num_edges;

      NodeWeight max_node_weight = 0;
      for (const auto &local_max_node_weight : local_max_node_weight_ets) {
        max_node_weight = std::max(local_max_node_weight[b], max_node_weight);
      }
      local_max_node_weight[b] = max_node_weight;

      NodeWeight total_node_weight = 0;
      for (const auto &local_total_node_weight : local_total_node_weight_ets) {
        total_node_weight += local_total_node_weight[b];
      }
      local_total_node_weight[b] = total_node_weight;

      EdgeWeight total_edge_weight = 0;
      for (const auto &local_total_edge_weight : local_total_edge_weight_ets) {
        total_node_weight += local_total_edge_weight[b];
      }
      local_total_edge_weight[b] = total_edge_weight;
    });
  };

  StaticArray<NodeID> global_to_local(num_nodes, static_array::noinit);
  StaticArray<NodeID> shared_local_to_global(num_nodes, static_array::noinit);
  StaticArray<NodeID> local_to_global_offsets(num_blocks);

  TIMED_SCOPE("Fill mappings") {
    parallel::prefix_sum(
        local_num_nodes.begin(), local_num_nodes.end() - 1, local_to_global_offsets.begin() + 1
    );

    std::vector<NodeID> local_to_global_size(num_blocks);
    tbb::parallel_for(tbb::blocked_range<NodeID>(0, num_nodes), [&](auto &local_nodes) {
      const NodeID first_local_node = local_nodes.begin();
      const NodeID last_local_node = local_nodes.end();

      for (NodeID u = first_local_node; u < last_local_node; ++u) {
        const BlockID u_block = p_graph.block(u);
        const NodeID u_local =
            __atomic_fetch_add(&local_to_global_size[u_block], 1, __ATOMIC_RELAXED);

        global_to_local[u] = u_local;
        shared_local_to_global[local_to_global_offsets[u_block] + u_local] = u;
      }
    });
  };

  ScalableVector<shm::Graph> subgraphs(num_blocks);
  TIMED_SCOPE("Construct graph objects") {
    tbb::parallel_for<BlockID>(0, num_blocks, [&](const BlockID b) {
      subgraphs[b] = shm::Graph(std::make_unique<BlockInducedSubgraph<Graph>>(
          &graph,
          StaticArray<BlockID>(p_graph.raw_partition().size(), p_graph.raw_partition().data()),
          local_num_nodes[b],
          local_num_edges[b],
          local_max_node_weight[b],
          local_total_node_weight[b],
          local_total_edge_weight[b],
          b,
          StaticArray<NodeID>(global_to_local.size(), global_to_local.data()),
          LocalToGlobalMapping(
              true,
              local_to_global_offsets[b],
              shared_local_to_global.data(),
              backup_local_to_global
          ),
          StaticArray<std::uint8_t>(border_nodes.size(), border_nodes.data())
      ));
    });
  };

  return {
      std::move(subgraphs),
      std::move(global_to_local),
      std::move(shared_local_to_global),
      std::move(border_nodes)
  };
}
} // namespace

SpanSubgraphExtractionResult
extract_span_subgraphs(PartitionedGraph &p_graph, BlockID input_k, NodeID *backup_local_to_global) {
  return p_graph.reified([&](const auto &graph) {
    return extract_span_subgraphs_generic_graph(p_graph, graph, input_k, backup_local_to_global);
  });
}

PartitionedGraph copy_subgraph_partitions(
    PartitionedGraph p_graph,
    const ScalableVector<StaticArray<BlockID>> &p_subgraph_partitions,
    const BlockID k_prime,
    const BlockID input_k,
    const StaticArray<NodeID> &mapping
) {
  // The offset calculation works as follows:
  //
  // - while we have fewer blocks than `input_k`, each block is partitioned into the same number of
  //   sub-blocks -- thus, we can keep the default values specified above.
  //
  // - once we have extended the number of blocks to `input_k`, blocks may have been partitioned
  //   into differing numbers of sub-blocks
  std::vector<BlockID> k0(p_graph.k() + 1, k_prime / p_graph.k());
  if (k_prime == input_k) {
    for (const BlockID b : p_graph.blocks()) {
      k0[b + 1] = partitioning::compute_final_k(b, p_graph.k(), input_k);
    }
  }

  k0.front() = 0;
  parallel::prefix_sum(k0.begin(), k0.end(), k0.begin());

  DBG << "Copying partition after recursive bipartitioning: extended " << p_graph.k()
      << "-way partition to " << k_prime << "-way, goal: " << input_k
      << " with block offsets: " << k0;

  StaticArray<BlockID> partition = p_graph.take_raw_partition();
  p_graph.pfor_nodes([&](const NodeID u) {
    const BlockID b = partition[u];
    const NodeID s_u = mapping[u];
    partition[u] = k0[b] + p_subgraph_partitions[b][s_u];
  });

  PartitionedGraph new_p_graph(p_graph.graph(), k_prime, std::move(partition));
  DBG << "Statistics after copying the subgraph partitions:";
  DBG << "  Block weights: " << new_p_graph.raw_block_weights();
  DBG << "  Cut:           " << metrics::edge_cut(new_p_graph);
  DBG << "  Imbalance:     " << metrics::imbalance(new_p_graph);

  return new_p_graph;
}
} // namespace kaminpar::shm::graph
