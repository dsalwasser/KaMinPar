/*******************************************************************************
 * Multi-way flow refiner.
 *
 * @file:   multiway_flow_refiner.cc
 * @author: Daniel Salwasser
 * @date:   10.04.2025
 ******************************************************************************/
#include "kaminpar-shm/refinement/flow/multiway_flow_refiner.h"

#include <algorithm>
#include <limits>
#include <queue>
#include <span>
#include <unordered_set>
#include <utility>
#include <vector>

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/metrics.h"
#include "kaminpar-shm/refinement/flow/multiway_cut/isolating_cut_heuristic.h"
#include "kaminpar-shm/refinement/flow/multiway_cut/labelling_function_heuristic.h"
#include "kaminpar-shm/refinement/flow/rebalancer/dynamic_greedy_balancer.h"
#include "kaminpar-shm/refinement/flow/rebalancer/static_greedy_balancer.h"
#include "kaminpar-shm/refinement/flow/util/border_region.h"
#include "kaminpar-shm/refinement/flow/util/reverse_edge_index.h"
#include "kaminpar-shm/refinement/flow/util/terminal_sets.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/scalable_vector.h"
#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

class MultipartitionFlowRefiner {
  SET_DEBUG(true);

  struct FlowNetwork {
    CSRGraph graph;
    StaticArray<NodeID> reverse_edges;
    std::unordered_map<NodeID, NodeID> global_to_local_mapping;
  };

  struct Cut {
    NodeWeight weight;
    std::unordered_set<NodeID> nodes;
  };

  struct Move {
    NodeID node;
    BlockID old_block;
    BlockID new_block;
  };

  struct Result {
    EdgeWeight gain;
    ScalableVector<Move> moves;

    Result() : gain(0) {};
    Result(EdgeWeight gain, ScalableVector<Move> moves) : gain(gain), moves(std::move(moves)) {};
  };

  [[nodiscard]] static EdgeWeight
  compute_cut_value(const CSRGraph &graph, std::span<const BlockID> partition) {
    EdgeWeight cut_value = 0;

    for (const NodeID u : graph.nodes()) {
      graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
        cut_value += (partition[u] != partition[v]) ? w : 0;
      });
    }

    return cut_value / 2;
  }

public:
  MultipartitionFlowRefiner(
      const PartitionContext &p_ctx,
      const MultiwayFlowRefinementContext &f_ctx,
      const PartitionedCSRGraph &p_graph,
      const CSRGraph &graph
  )
      : _p_ctx(p_ctx),
        _f_ctx(f_ctx),
        _p_graph(p_graph),
        _graph(graph),
        _dynamic_balancer(p_ctx.max_block_weights()) {

    switch (_f_ctx.cut_algorithm) {
    case CutAlgorithm::ISOLATING_CUT_HEURISTIC:
      _multiway_cut_algorithm =
          std::make_unique<IsolatingCutHeuristic>(_f_ctx.isolating_cut_heuristic);
      break;
    case CutAlgorithm::LABELLING_FUNCTION_HEURISTIC:
      _multiway_cut_algorithm =
          std::make_unique<LabellingFunctionHeuristic>(_f_ctx.labelling_function_heuristic);
      break;
    }

    if (_f_ctx.unconstrained) {
      _p_graph_rebalancing_copy =
          PartitionedCSRGraph(PartitionedCSRGraph::seq(), graph, p_graph.k());
    }
  }

  Result refine(const EdgeWeight global_cut_value) {
    SCOPED_TIMER("Refine Partition");

    _global_cut_value = global_cut_value;
    _constrained_cut_value = global_cut_value;
    _unconstrained_cut_value = global_cut_value;

    compute_border_regions();
    for (BorderRegion &border_region : _border_regions) {
      expand_border_region(border_region);
    }
    construct_flow_network();
    run_refinement();

    if (_unconstrained_cut_value < _constrained_cut_value) {
      const EdgeWeight gain = _global_cut_value - _unconstrained_cut_value;
      return Result(gain, std::move(_unconstrained_moves));
    } else {
      const EdgeWeight gain = _global_cut_value - _constrained_cut_value;
      return Result(gain, std::move(_constrained_moves));
    }
  }

private:
  void compute_border_regions() {
    SCOPED_TIMER("Compute Border Regions");

    _border_regions.clear();
    _border_regions.reserve(_p_graph.k());

    for (BlockID block = 0; block < _p_graph.k(); ++block) {
      const BlockWeight max_border_region_weight =
          _f_ctx.border_region_scaling_factor * _p_graph.block_weight(block);
      _border_regions.emplace_back(block, max_border_region_weight);
    }

    for (const NodeID u : _graph.nodes()) {
      const BlockID u_block = _p_graph.block(u);

      BorderRegion &u_border_region = _border_regions[u_block];
      if (u_border_region.contains(u)) {
        continue;
      }

      const NodeWeight u_weight = _graph.node_weight(u);
      if (!u_border_region.fits(u_weight)) {
        continue;
      }

      bool is_border_region_node = false;
      _graph.adjacent_nodes(u, [&](const NodeID v) {
        const BlockID v_block = _p_graph.block(v);
        if (u_block == v_block) {
          return;
        }

        BorderRegion &v_border_region = _border_regions[v_block];
        if (v_border_region.contains(v)) {
          is_border_region_node = true;
          return;
        }

        const NodeWeight v_weight = _graph.node_weight(v);
        if (v_border_region.fits(v_weight)) {
          v_border_region.insert(v, v_weight);
          is_border_region_node = true;
        }
      });

      if (is_border_region_node) {
        u_border_region.insert(u, u_weight);
      }
    }
  }

  void expand_border_region(BorderRegion &border_region) const {
    SCOPED_TIMER("Expand Border Region");

    std::queue<std::pair<NodeID, NodeID>> bfs_queue;
    for (const NodeID u : border_region.nodes()) {
      bfs_queue.emplace(u, 0);
    }

    const BlockID block = border_region.block();
    while (!bfs_queue.empty()) {
      const auto [u, u_distance] = bfs_queue.front();
      bfs_queue.pop();

      _graph.adjacent_nodes(u, [&](const NodeID v) {
        if (_p_graph.block(v) != block || border_region.contains(v)) {
          return;
        }

        const NodeWeight v_weight = _graph.node_weight(v);
        if (border_region.fits(v_weight)) {
          border_region.insert(v, v_weight);

          if (u_distance < _f_ctx.max_border_distance) {
            bfs_queue.emplace(v, u_distance + 1);
          }
        }
      });
    }
  }

  void construct_flow_network() {
    SCOPED_TIMER("Construct Flow Network");

    const BlockID num_blocks = _p_graph.k();
    const NodeID first_node_id = num_blocks;
    NodeID cur_node = first_node_id;

    std::unordered_map<NodeID, NodeID> global_to_local_mapping;
    for (const BorderRegion &border_region : _border_regions) {
      for (const NodeID u : border_region.nodes()) {
        global_to_local_mapping.emplace(u, cur_node++);
      }
    }

    const NodeID num_nodes = cur_node;
    StaticArray<EdgeID> nodes(num_nodes + 1, static_array::noinit);
    StaticArray<NodeWeight> node_weights(num_nodes, static_array::noinit);

    std::vector<bool> adjacency(num_blocks);
    std::vector<EdgeID> num_terminal_edges(num_blocks, 0);

    cur_node = first_node_id;
    for (BlockID block = 0; block < num_blocks; ++block) {
      for (const NodeID u : _border_regions[block].nodes()) {
        std::fill(adjacency.begin(), adjacency.end(), false);

        EdgeID num_neighbors = 0;
        _graph.adjacent_nodes(u, [&](const NodeID v) {
          if (global_to_local_mapping.contains(v)) {
            num_neighbors += 1;
            return;
          }

          const BlockID v_block = _p_graph.block(v);
          adjacency[v_block] = true;
        });

        for (BlockID block = 0; block < num_blocks; ++block) {
          if (adjacency[block]) {
            num_neighbors += 1;
            num_terminal_edges[block] += 1;
          }
        }

        const NodeWeight u_weight = _graph.node_weight(u);
        nodes[cur_node] = num_neighbors;
        node_weights[cur_node] = u_weight;

        cur_node += 1;
      }
    }

    for (BlockID block = 0; block < num_blocks; ++block) {
      nodes[block] = num_terminal_edges[block];
      node_weights[block] = _p_graph.block_weight(block) - _border_regions[block].weight();
    }

    nodes.back() = 0;
    std::partial_sum(nodes.begin(), nodes.end(), nodes.begin());

    const EdgeID num_edges = nodes.back();
    StaticArray<NodeID> edges(num_edges, static_array::noinit);
    StaticArray<EdgeWeight> edge_weights(num_edges, static_array::noinit);
    StaticArray<NodeID> reverse_edges(num_edges, static_array::noinit);

    EdgeWeight cut_value = 0;
    std::vector<EdgeWeight> terminal_edge_weight(num_blocks, 0);

    cur_node = first_node_id;
    for (BlockID block = 0; block < num_blocks; ++block) {
      for (const NodeID u : _border_regions[block].nodes()) {
        const NodeID u_local = cur_node;

        std::fill(adjacency.begin(), adjacency.end(), false);
        std::fill(terminal_edge_weight.begin(), terminal_edge_weight.end(), 0);

        EdgeID u_edge = nodes[u_local];
        _graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
          if (auto it = global_to_local_mapping.find(v); it != global_to_local_mapping.end()) {
            const NodeID v_local = it->second;
            if (u_local >= v_local) {
              return;
            }

            const BlockID v_block = _p_graph.block(v);
            cut_value += (block != v_block) ? w : 0;

            u_edge -= 1;
            edges[u_edge] = v_local;
            edge_weights[u_edge] = w;

            const EdgeID v_edge = --nodes[v_local];
            edges[v_edge] = u_local;
            edge_weights[v_edge] = w;

            reverse_edges[u_edge] = v_edge;
            reverse_edges[v_edge] = u_edge;
            return;
          }

          const BlockID v_block = _p_graph.block(v);
          adjacency[v_block] = true;
          terminal_edge_weight[v_block] += w;
        });

        for (BlockID adjcent_block = 0; adjcent_block < num_blocks; ++adjcent_block) {
          if (adjacency[adjcent_block]) {
            const EdgeWeight w = terminal_edge_weight[adjcent_block];
            cut_value += (block != adjcent_block) ? w : 0;

            u_edge -= 1;
            edges[u_edge] = adjcent_block;
            edge_weights[u_edge] = w;

            const EdgeID terminal_edge = --nodes[adjcent_block];
            edges[terminal_edge] = cur_node;
            edge_weights[terminal_edge] = w;

            reverse_edges[u_edge] = terminal_edge;
            reverse_edges[terminal_edge] = u_edge;
          }
        }

        nodes[u_local] = u_edge;
        cur_node += 1;
      }
    }

    CSRGraph graph(
        CSRGraph::seq(),
        std::move(nodes),
        std::move(edges),
        std::move(node_weights),
        std::move(edge_weights)
    );
    KASSERT(debug::validate_graph(graph), "constructed invalid flow network", assert::heavy);
    KASSERT(
        debug::is_valid_reverse_edge_index(graph, reverse_edges),
        "constructed an invalid reverse edge index",
        assert::heavy
    );

    _initial_cut_value = cut_value;
    _flow_network =
        FlowNetwork(std::move(graph), std::move(reverse_edges), std::move(global_to_local_mapping));

    _terminal_sets.initialize(num_blocks, _flow_network.graph.n());
    for (BlockID block = 0; block < num_blocks; ++block) {
      _terminal_sets.set_terminal_set(block, block);
    }

    _local_p_graph =
        PartitionedCSRGraph(PartitionedCSRGraph::seq(), _flow_network.graph, _p_graph.k());
    for (BlockID block = 0; block < num_blocks; ++block) {
      _local_p_graph.set_block(block, block);
    }
    for (const auto &[u, u_local] : _flow_network.global_to_local_mapping) {
      _local_p_graph.set_block(u_local, _p_graph.block(u));
    }
  }

  void run_refinement() {
    if (_f_ctx.unconstrained) {
      initialize_rebalancer();
    }

    DBG << "Starting refinement with an initial cut of " << _global_cut_value;
    while (true) {
      const auto [cut_value, cut_edges] = TIMED_SCOPE("Compute Multi-Way Cut") {
        return _multiway_cut_algorithm->compute(
            _local_p_graph, _flow_network.graph, _flow_network.reverse_edges, _terminal_sets
        );
      };

      const EdgeWeight gain = _initial_cut_value - cut_value;
      const EdgeWeight new_global_cut_value = _global_cut_value - gain;
      DBG << "Found a cut with gain " << gain << " (" << _global_cut_value << " -> "
          << new_global_cut_value << ")";

      update_partition(cut_edges);
      KASSERT(cut_value == metrics::edge_cut_seq(_local_p_graph));

      const auto [is_balanced, lightest_block] = block_weights_status();
      if (is_balanced) {
        DBG << "Found cut is a balanced";
        _constrained_cut_value = new_global_cut_value;

        compute_moves();
        break;
      }

      if (_f_ctx.unconstrained) {
        rebalance(new_global_cut_value);

        if (_f_ctx.abort_on_candidate_cut && _unconstrained_cut_value < _global_cut_value &&
            _unconstrained_cut_value < new_global_cut_value) {
          break;
        }
      }

      const bool found_new_terminal_node = add_terminal(lightest_block);
      if (!found_new_terminal_node) {
        LOG_WARNING << "Failed to find a suitable new terminal node";
        break;
      }
    }
  }

  void update_partition(const std::unordered_set<NodeID> &cut_edges) {
    SCOPED_TIMER("Update Partition");

    for (BlockID block = 0; block < _terminal_sets.num_terminal_sets(); ++block) {
      Cut cut = compute_cut_nodes(_terminal_sets.terminal_set_nodes(block), cut_edges);
      DBG << "Block " << block << " has weight " << cut.weight << "/"
          << _p_ctx.max_block_weight(block);

      for (const NodeID u : cut.nodes) {
        _local_p_graph.set_block(u, block);
      }
    }
  }

  Cut compute_cut_nodes(
      std::span<const NodeID> terminals, const std::unordered_set<EdgeID> &cut_edges
  ) {
    SCOPED_TIMER("Compute Cut Nodes");
    const CSRGraph &graph = _flow_network.graph;

    NodeWeight cut_weight = 0;
    std::unordered_set<NodeID> cut_nodes;

    std::queue<NodeID> bfs_queue;
    for (const NodeID terminal : terminals) {
      cut_weight += graph.node_weight(terminal);
      cut_nodes.insert(terminal);
      bfs_queue.push(terminal);
    }

    while (!bfs_queue.empty()) {
      const NodeID u = bfs_queue.front();
      bfs_queue.pop();

      graph.neighbors(u, [&](const EdgeID e, const NodeID v) {
        if (cut_nodes.contains(v) || cut_edges.contains(e)) {
          return;
        }

        cut_weight += graph.node_weight(v);
        cut_nodes.insert(v);
        bfs_queue.push(v);
      });
    }

    return Cut(cut_weight, std::move(cut_nodes));
  }

  [[nodiscard]] std::pair<bool, BlockWeight> block_weights_status() const {
    bool is_balanced = true;
    BlockID lightest_block = kInvalidBlockID;

    BlockWeight lightest_block_weight = std::numeric_limits<BlockWeight>::max();
    for (const BlockID block : _local_p_graph.blocks()) {
      const BlockWeight block_weight = _local_p_graph.block_weight(block);

      if (block_weight > _p_ctx.max_block_weight(block)) {
        is_balanced = false;
      }

      if (block_weight < lightest_block_weight) {
        lightest_block = block;
        lightest_block_weight = block_weight;
      }
    }

    KASSERT(lightest_block != kInvalidBlockID);
    return {is_balanced, lightest_block};
  }

  bool add_terminal(const BlockID block) {
    SCOPED_TIMER("Add Terminal");

    // TODO

    return false;
  }

  void compute_moves() {
    SCOPED_TIMER("Compute Moves");

    _constrained_moves.clear();
    for (const auto &[u, u_local] : _flow_network.global_to_local_mapping) {
      const BlockID old_block = _p_graph.block(u);
      const BlockID new_block = _local_p_graph.block(u_local);

      if (old_block != new_block) {
        _constrained_moves.emplace_back(u, old_block, new_block);
      }
    }
  }

  void initialize_rebalancer() {
    SCOPED_TIMER("Initialize Rebalancer");

    for (const NodeID u : _graph.nodes()) {
      _p_graph_rebalancing_copy.set_block(u, _p_graph.block(u));
    }

    _dynamic_balancer.setup(_p_graph_rebalancing_copy, _graph);
  }

  void rebalance(const EdgeWeight cut_value) {
    TIMED_SCOPE("Initialize Partitioned Graph") {
      for (const auto [u, u_local] : _flow_network.global_to_local_mapping) {
        _p_graph_rebalancing_copy.set_block(u, _local_p_graph.block(u_local));
      }
    };

    KASSERT(
        metrics::edge_cut_seq(_p_graph_rebalancing_copy) == cut_value,
        "Given an incorrect cut value for partitioned graph",
        assert::heavy
    );

    const auto [balanced, gain, moved_nodes] = TIMED_SCOPE("Rebalance") {
      return _dynamic_balancer.rebalance();
    };

    if (!balanced) {
      DBG << "Rebalancing failed to produce a balanced cut";
    } else {
      KASSERT(
          metrics::is_balanced(_p_graph_rebalancing_copy, _p_ctx),
          "Rebalancing resulted in an inbalanced partition",
          assert::heavy
      );
      SCOPED_TIMER("Compute Moves");

      const EdgeWeight rebalanced_cut_value = cut_value - gain;
      DBG << "Rebalanced imbalanced cut with resulting global value " << rebalanced_cut_value;

      KASSERT(
          metrics::edge_cut_seq(_p_graph_rebalancing_copy) == rebalanced_cut_value,
          "Given an incorrect cut value for rebalanced partitioned graph",
          assert::heavy
      );

      if (rebalanced_cut_value < _unconstrained_cut_value) {
        _unconstrained_cut_value = rebalanced_cut_value;
        _unconstrained_moves.clear();

        for (const auto [u, _] : _flow_network.global_to_local_mapping) {
          const BlockID old_block = _p_graph.block(u);
          const BlockID new_block = _p_graph_rebalancing_copy.block(u);

          if (old_block != new_block) {
            _unconstrained_moves.emplace_back(u, old_block, new_block);
          };
        }

        for (const NodeID u : moved_nodes) {
          if (_flow_network.global_to_local_mapping.contains(u)) {
            continue;
          }

          const BlockID old_block = _p_graph.block(u);
          const BlockID new_block = _p_graph_rebalancing_copy.block(u);
          if (old_block != new_block) {
            _unconstrained_moves.emplace_back(u, old_block, new_block);
          };
        }
      }
    }

    TIMED_SCOPE("Reset Partitioned Graph") {
      for (const NodeID u : moved_nodes) {
        _p_graph_rebalancing_copy.set_block(u, _p_graph.block(u));
      }
    };
  }

private:
  const PartitionContext &_p_ctx;
  const MultiwayFlowRefinementContext &_f_ctx;

  const PartitionedCSRGraph &_p_graph;
  const CSRGraph &_graph;

  ScalableVector<BorderRegion> _border_regions;
  FlowNetwork _flow_network;
  TerminalSets _terminal_sets;

  EdgeWeight _global_cut_value;
  EdgeWeight _initial_cut_value;
  PartitionedCSRGraph _local_p_graph;

  EdgeWeight _constrained_cut_value;
  ScalableVector<Move> _constrained_moves;

  EdgeWeight _unconstrained_cut_value;
  ScalableVector<Move> _unconstrained_moves;

  std::unique_ptr<MultiwayCutAlgorithm> _multiway_cut_algorithm;

  PartitionedCSRGraph _p_graph_rebalancing_copy;
  DynamicGreedyMultiBalancer<PartitionedCSRGraph, CSRGraph> _dynamic_balancer;
};

MultiwayFlowRefiner::MultiwayFlowRefiner(const Context &ctx)
    : _f_ctx(ctx.refinement.multiway_flow) {}

MultiwayFlowRefiner::~MultiwayFlowRefiner() = default;

std::string MultiwayFlowRefiner::name() const {
  return "Multi-Way Flow Refinement";
}

void MultiwayFlowRefiner::initialize([[maybe_unused]] const PartitionedGraph &p_graph) {}

bool MultiwayFlowRefiner::refine(
    PartitionedGraph &p_graph, [[maybe_unused]] const PartitionContext &p_ctx
) {
  return reified(
      p_graph,
      [&](const auto &csr_graph) {
        // The partition refiner works with PartitionedCSRGraph instead of PartitionedGraph.
        // Intead of copying the partition, we use a span to access the partition.
        StaticArray<BlockID> &partition = p_graph.raw_partition();
        StaticArray<BlockID> partition_span(partition.size(), partition.data());

        StaticArray<BlockWeight> &block_weights = p_graph.raw_block_weights();
        StaticArray<BlockWeight> block_weights_span(block_weights.size(), block_weights.data());

        PartitionedCSRGraph p_csr_graph(
            csr_graph, p_graph.k(), std::move(partition_span), std::move(block_weights_span)
        );
        return refine(p_csr_graph, csr_graph, p_ctx);
      },
      [&]([[maybe_unused]] const auto &compressed_graph) {
        LOG_WARNING << "Cannot refine a compressed graph using the multiway flow refiner.";
        return false;
      }
  );
}

bool MultiwayFlowRefiner::refine(
    PartitionedCSRGraph &p_graph, const CSRGraph &graph, const PartitionContext &p_ctx
) {
  SCOPED_TIMER("Multi-Way Flow Refinement");
  SCOPED_HEAP_PROFILER("Multi-Way Flow Refinement");

  const EdgeWeight initial_cut_value = metrics::edge_cut(p_graph);
  MultipartitionFlowRefiner refiner(p_ctx, _f_ctx, p_graph, graph);

  std::size_t num_round = 0;
  EdgeWeight cut_value = initial_cut_value;
  while (cut_value > 0) {
    num_round += 1;
    DBG << "Starting round " << num_round;

    const auto [gain, moves] = refiner.refine(cut_value);
    const EdgeWeight new_cut_value = cut_value - gain;

    DBG << "Found balanced cut with gain " << gain << " (" << cut_value << " -> " << new_cut_value
        << ")";

    if (gain > 0) {
      cut_value = new_cut_value;

      TIMED_SCOPE("Apply Moves") {
        for (const auto &move : moves) {
          KASSERT(
              p_graph.block(move.node) == move.old_block,
              "Move sequence contains invalid old block ids",
              assert::heavy
          );

          p_graph.set_block(move.node, move.new_block);
        }
      };
    }

    KASSERT(
        metrics::edge_cut_seq(p_graph) == new_cut_value, "Computed an invalid gain", assert::heavy
    );

    KASSERT(
        metrics::is_balanced(p_graph, p_ctx), "Computed an imbalanced move sequence", assert::heavy
    );

    const double relative_improvement = gain / static_cast<double>(cut_value);
    if (num_round == _f_ctx.max_num_rounds ||
        relative_improvement < _f_ctx.min_round_improvement_factor) {
      break;
    }
  }

  const bool found_improvement = cut_value < initial_cut_value;
  return found_improvement;
}

} // namespace kaminpar::shm
