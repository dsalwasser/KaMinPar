/*******************************************************************************
 * Two-way flow refiner.
 *
 * @file:   twoway_flow_refiner.h
 * @author: Daniel Salwasser
 * @date:   10.04.2025
 ******************************************************************************/
#include "kaminpar-shm/refinement/flow/twoway_flow_refiner.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
#include <queue>
#include <span>
#include <unordered_set>
#include <utility>
#include <vector>

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/metrics.h"
#include "kaminpar-shm/refinement/flow/max_flow/edmond_karp_algorithm.h"
#include "kaminpar-shm/refinement/flow/max_flow/max_flow_algorithm.h"
#include "kaminpar-shm/refinement/flow/max_flow/preflow_push_algorithm.h"
#include "kaminpar-shm/refinement/flow/piercing/piercing_heuristic.h"
#include "kaminpar-shm/refinement/flow/util/border_region.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

class BipartitionFlowRefiner {
  SET_DEBUG(false);

  struct FlowNetwork {
    NodeID source;
    NodeID sink;

    CSRGraph graph;
    std::unordered_map<NodeID, NodeID> global_to_local_mapping;
  };

  struct Cut {
    NodeWeight weight;
    std::unordered_set<NodeID> nodes;
  };

public:
  struct Move {
    NodeID node;
    BlockID old_block;
    BlockID new_block;
  };

public:
  BipartitionFlowRefiner(
      const PartitionContext &p_ctx,
      const TwowayFlowRefinementContext &f_ctx,
      const PartitionedGraph &p_graph,
      const CSRGraph &graph
  )
      : _p_ctx(p_ctx),
        _f_ctx(f_ctx),
        _p_graph(p_graph),
        _graph(graph) {
    switch (_f_ctx.flow_algorithm) {
    case FlowAlgorithm::EDMONDS_KARP:
      _max_flow_algorithm = std::make_unique<EdmondsKarpAlgorithm>();
      break;
    case FlowAlgorithm::PREFLOW_PUSH:
      _max_flow_algorithm = std::make_unique<PreflowPushAlgorithm>(_f_ctx.preflow_push);
      break;
    }
  }

  std::vector<Move> refine(const BlockID block1, const BlockID block2) {
    auto [border_region1, border_region2] = compute_border_regions(block1, block2);
    expand_border_region(border_region1);
    expand_border_region(border_region2);

    FlowNetwork flow_network = construct_flow_network(border_region1, border_region2);
    if (flow_network.sink == kInvalidNodeID || flow_network.source == kInvalidNodeID) {
      LOG_WARNING << "Border region contains all nodes inside block; "
                     "aborting refinement for block pair "
                  << block1 << " and " << block2;
      return std::vector<Move>();
    }

    border_region1.project(flow_network.global_to_local_mapping);
    border_region2.project(flow_network.global_to_local_mapping);
    PiercingHeuristic piercing_heuristic(
        flow_network.graph, border_region1.nodes(), border_region2.nodes()
    );

    StaticArray<EdgeWeight> flow(flow_network.graph.m());
    std::unordered_set<NodeID> source_side_nodes{flow_network.source};
    std::unordered_set<NodeID> sink_side_nodes{flow_network.sink};

    const NodeWeight total_weight = _p_graph.block_weight(block1) + _p_graph.block_weight(block2);
    const NodeWeight max_block1_weight = _p_ctx.max_block_weight(block1);
    const NodeWeight max_block2_weight = _p_ctx.max_block_weight(block2);

    while (true) {
      TIMED_SCOPE("Compute Max Flow") {
        _max_flow_algorithm->compute(flow_network.graph, source_side_nodes, sink_side_nodes, flow);
      };

      Cut source_cut = compute_source_cut(flow_network.graph, source_side_nodes, flow);
      Cut sink_cut = compute_sink_cut(flow_network.graph, sink_side_nodes, flow);
      KASSERT(
          debug::are_terminals_disjoint(source_cut.nodes, sink_cut.nodes),
          "source and sink nodes are not disjoint",
          assert::heavy
      );

      const NodeWeight source_cut_weight_prime = total_weight - source_cut.weight;
      const bool is_source_cut_balanced =
          source_cut.weight <= max_block1_weight && source_cut_weight_prime <= max_block2_weight;

      if (is_source_cut_balanced) {
        DBG << "Found balanced source-side cut";

        return compute_moves(
            flow_network.global_to_local_mapping,
            border_region1.nodes(),
            source_cut.nodes,
            block1,
            block2
        );
      }

      const NodeWeight sink_cut_weight_prime = total_weight - sink_cut.weight;
      const bool is_sink_cut_balanced =
          sink_cut.weight <= max_block2_weight && sink_cut_weight_prime <= max_block1_weight;

      if (is_sink_cut_balanced) {
        DBG << "Found balanced sink-side cut";

        return compute_moves(
            flow_network.global_to_local_mapping,
            border_region2.nodes(),
            sink_cut.nodes,
            block2,
            block1
        );
      }

      SCOPED_TIMER("Compute Piercing Node");
      if (source_cut.weight <= sink_cut.weight) {
        DBG << "Piercing on source-side (" << source_cut.weight << "/" << max_block1_weight << ", "
            << source_cut_weight_prime << "/" << max_block2_weight << ")";

        const NodeWeight max_piercing_node_weight = max_block1_weight - source_cut.weight;
        const NodeID piercing_node = piercing_heuristic.pierce_on_source_side(
            source_cut.nodes, sink_cut.nodes, max_piercing_node_weight
        );

        if (piercing_node == kInvalidNodeID) {
          LOG_WARNING << "Failed to find a suitable pierce node; "
                         "aborting refinement for block pair "
                      << block1 << " and " << block2;
          return std::vector<Move>();
        }

        source_side_nodes = std::move(source_cut.nodes);
        source_side_nodes.insert(piercing_node);

        if (sink_side_nodes.contains(piercing_node)) {
          sink_side_nodes.erase(piercing_node);
        }
      } else {
        DBG << "Piercing on sink-side (" << sink_cut.weight << "/" << max_block2_weight << ", "
            << sink_cut_weight_prime << "/" << max_block1_weight << ")";

        const NodeWeight max_piercing_node_weight = max_block2_weight - sink_cut.weight;
        const NodeID piercing_node = piercing_heuristic.pierce_on_sink_side(
            source_cut.nodes, sink_cut.nodes, max_piercing_node_weight
        );

        if (piercing_node == kInvalidNodeID) {
          LOG_WARNING << "Failed to find a suitable pierce node; "
                         "aborting refinement for block pair "
                      << block1 << " and " << block2;
          return std::vector<Move>();
        }

        sink_side_nodes = std::move(sink_cut.nodes);
        sink_side_nodes.insert(piercing_node);

        if (source_side_nodes.contains(piercing_node)) {
          source_side_nodes.erase(piercing_node);
        }
      }

      KASSERT(
          debug::are_terminals_disjoint(source_cut.nodes, sink_cut.nodes),
          "source and sink nodes are not disjoint",
          assert::heavy
      );
    }
  }

private:
  std::pair<BorderRegion, BorderRegion>
  compute_border_regions(const BlockID block1, const BlockID block2) {
    SCOPED_TIMER("Compute Border Regions");

    const NodeWeight max_border_region_weight1 = std::min<NodeWeight>(
        _f_ctx.border_region_scaling_factor * _p_ctx.max_block_weight(block2) -
            _p_graph.block_weight(block2),
        _p_graph.block_weight(block1)
    );
    BorderRegion border_region1(block1, max_border_region_weight1);

    const NodeWeight max_border_region_weight2 = std::min<NodeWeight>(
        _f_ctx.border_region_scaling_factor * _p_ctx.max_block_weight(block1) -
            _p_graph.block_weight(block1),
        _p_graph.block_weight(block2)
    );
    BorderRegion border_region2(block2, max_border_region_weight2);

    for (NodeID u : _graph.nodes()) {
      if (_p_graph.block(u) != block1) {
        continue;
      }

      const NodeWeight u_weight = _graph.node_weight(u);
      if (!border_region1.fits(u_weight)) {
        continue;
      }

      bool is_border_region_node = false;
      _graph.adjacent_nodes(u, [&](const NodeID v) {
        if (_p_graph.block(v) != block2) {
          return;
        }

        if (border_region2.contains(v)) {
          is_border_region_node = true;
          return;
        }

        const NodeWeight v_weight = _graph.node_weight(v);
        if (border_region2.fits(v_weight)) {
          is_border_region_node = true;
          border_region2.insert(v, v_weight);
        }
      });

      if (is_border_region_node) {
        border_region1.insert(u, u_weight);
      }
    }

    return {std::move(border_region1), std::move(border_region2)};
  }

  void expand_border_region(BorderRegion &border_region) {
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

  FlowNetwork
  construct_flow_network(const BorderRegion &border_region1, const BorderRegion &border_region2) {
    SCOPED_TIMER("Construct Flow Network");

    NodeID kSource = 0;
    NodeID kSink = 1;
    constexpr NodeID kFirstNodeID = 2;

    NodeID cur_node = kFirstNodeID;

    std::unordered_map<NodeID, NodeID> global_to_local_mapping;
    for (const BorderRegion &border_region :
         {std::cref(border_region1), std::cref(border_region2)}) {
      for (const NodeID u : border_region.nodes()) {
        global_to_local_mapping.emplace(u, cur_node++);
      }
    }

    const NodeID num_nodes = 2 + border_region1.num_nodes() + border_region2.num_nodes();
    StaticArray<EdgeID> nodes(num_nodes + 1, static_array::noinit);
    StaticArray<NodeWeight> node_weights(num_nodes, static_array::noinit);

    cur_node = kFirstNodeID;
    for (BlockID terminal = 0; terminal < 2; ++terminal) {
      const BorderRegion &border_region = (terminal == 0) ? border_region1 : border_region2;

      NodeWeight border_region_weight = 0;
      EdgeID num_terminal_edges = 0;
      for (const NodeID u : border_region.nodes()) {
        EdgeID num_neighbors = 0;
        _graph.adjacent_nodes(u, [&](const NodeID v) {
          num_neighbors += global_to_local_mapping.contains(v) ? 1 : 0;
        });

        const bool has_non_border_region_neighbor = num_neighbors != _graph.degree(u);
        if (has_non_border_region_neighbor) { // Node has an edge to its corresponding terminal
          num_neighbors += 1;
          num_terminal_edges += 1;
        }

        const NodeWeight u_weight = _graph.node_weight(u);
        nodes[cur_node + 1] = num_neighbors;
        node_weights[cur_node] = u_weight;

        border_region_weight += u_weight;
        cur_node += 1;
      }

      if (num_terminal_edges == 0) {
        if (terminal == 0) {
          kSource = kInvalidNodeID;
        } else {
          kSink = kInvalidNodeID;
        }
      }

      nodes[terminal + 1] = num_terminal_edges;
      node_weights[terminal] = _p_graph.block_weight(border_region.block()) - border_region_weight;
    }
    KASSERT(cur_node == num_nodes);

    nodes[0] = 0;
    std::partial_sum(nodes.begin(), nodes.end(), nodes.begin());

    const EdgeID num_edges = nodes.back();
    StaticArray<NodeID> edges(num_edges, static_array::noinit);
    StaticArray<EdgeWeight> edge_weights(num_edges, static_array::noinit);

    // To prevent integer overflow during max-flow computation use an upper bound
    // for the weights of edges connected to terminals. TOOD: improve solutions
    const EdgeWeight terminal_edge_weight = _graph.max_edge_weight() * _graph.n();

    cur_node = kFirstNodeID;
    EdgeID cur_edge = nodes[kFirstNodeID];
    EdgeID cur_source_edge = 0;
    for (BlockID terminal = 0; terminal < 2; ++terminal) {
      const BorderRegion &border_region = (terminal == 0) ? border_region1 : border_region2;

      for (const NodeID u : border_region.nodes()) {
        _graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
          if (auto it = global_to_local_mapping.find(v); it != global_to_local_mapping.end()) {
            const NodeID v_local = it->second;

            edges[cur_edge] = v_local;
            edge_weights[cur_edge] = w;
            cur_edge += 1;
          }
        });

        const NodeID u_degree = cur_edge - nodes[cur_node];
        const bool has_non_border_region_neighbor = u_degree != _graph.degree(u);
        if (has_non_border_region_neighbor) { // Connect node to its corresponding terminal
          edges[cur_edge] = terminal;
          edge_weights[cur_edge] = terminal_edge_weight;
          cur_edge += 1;

          edges[cur_source_edge] = cur_node;
          edge_weights[cur_source_edge] = terminal_edge_weight;
          cur_source_edge += 1;
        }

        cur_node += 1;
      }
    }
    KASSERT(cur_node == num_nodes);
    KASSERT(cur_edge == num_edges);
    KASSERT(cur_source_edge == nodes[kFirstNodeID]);

    CSRGraph graph(
        std::move(nodes), std::move(edges), std::move(node_weights), std::move(edge_weights)
    );
    KASSERT(debug::validate_graph(graph), "constructed invalid flow network", assert::heavy);

    return FlowNetwork(kSource, kSink, std::move(graph), std::move(global_to_local_mapping));
  }

  Cut compute_source_cut(
      const CSRGraph &graph,
      const std::unordered_set<NodeID> &sources,
      std::span<const EdgeWeight> flow
  ) {
    return compute_cut(graph, sources, flow, true);
  }

  Cut compute_sink_cut(
      const CSRGraph &graph,
      const std::unordered_set<NodeID> &sinks,
      std::span<const EdgeWeight> flow
  ) {
    return compute_cut(graph, sinks, flow, false);
  }

  Cut compute_cut(
      const CSRGraph &graph,
      const std::unordered_set<NodeID> &terminals,
      std::span<const EdgeWeight> flow,
      const bool source_side
  ) {
    SCOPED_TIMER("Compute Reachable Nodes");

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

      graph.neighbors(u, [&](const EdgeID e, const NodeID v, const EdgeWeight c) {
        if (cut_nodes.contains(v)) {
          return;
        }

        const EdgeWeight e_flow = flow[e];
        const bool has_residual_capacity = source_side ? (e_flow < c) : (-e_flow < c);
        if (has_residual_capacity) {
          cut_weight += graph.node_weight(v);
          cut_nodes.insert(v);
          bfs_queue.push(v);
        }
      });
    }

    return Cut(cut_weight, std::move(cut_nodes));
  }

  std::vector<Move> compute_moves(
      const std::unordered_map<NodeID, NodeID> &global_to_local_mapping,
      const std::unordered_set<NodeID> &initial_terminal_side_nodes,
      const std::unordered_set<NodeID> &terminal_side_nodes,
      const BlockID block,
      const BlockID other_block
  ) {
    SCOPED_TIMER("Compute Moves");

    std::vector<Move> assignments;

    for (const auto &[u, u_local] : global_to_local_mapping) {
      const BlockID old_block = initial_terminal_side_nodes.contains(u) ? block : other_block;
      const BlockID new_block = terminal_side_nodes.contains(u_local) ? block : other_block;
      assignments.emplace_back(u, old_block, new_block);
    }

    return assignments;
  }

private:
  const PartitionContext &_p_ctx;
  const TwowayFlowRefinementContext &_f_ctx;

  const PartitionedGraph &_p_graph;
  const CSRGraph &_graph;

  std::unique_ptr<MaxFlowAlgorithm> _max_flow_algorithm;
};

class SequentialBlockPairScheduler {
  SET_DEBUG(true);

  using Move = BipartitionFlowRefiner::Move;

public:
  SequentialBlockPairScheduler(
      const PartitionContext &p_ctx, const TwowayFlowRefinementContext &f_ctx
  )
      : _p_ctx(p_ctx),
        _f_ctx(f_ctx) {}

  bool refine(PartitionedGraph &p_graph, const CSRGraph &graph) {
    _p_graph = &p_graph;
    _graph = &graph;

    const BlockID k = p_graph.k();
    if (_active_blocks.size() < k) {
      _active_blocks.resize(k, static_array::noinit);
    }

    std::fill(_active_blocks.begin(), _active_blocks.end(), false);
    _block_pairs = {};
    _next_block_pairs = {};

    for (BlockID block2 = 0; block2 < k; ++block2) {
      for (BlockID block1 = 0; block1 < block2; ++block1) {
        _block_pairs.emplace(block1, block2);
      }
    }

    BipartitionFlowRefiner refiner(_p_ctx, _f_ctx, p_graph, graph);
    EdgeWeight prev_cut_value = metrics::edge_cut_seq(p_graph);

    std::size_t num_round = 0;
    bool found_improvement = false;
    while (true) {
      num_round += 1;
      DBG << "Starting round " << num_round;

      EdgeWeight cut_value = prev_cut_value;
      while (!_block_pairs.empty()) {
        const auto [block1, block2] = _block_pairs.front();
        _block_pairs.pop();

        DBG << "Scheduling block pair " << block1 << " and " << block2;
        std::vector<Move> moves = refiner.refine(block1, block2);
        apply_moves(moves);

        const EdgeWeight new_cut_value = metrics::edge_cut_seq(p_graph);
        const EdgeWeight gain = cut_value - new_cut_value;
        DBG << "Found balanced cut with gain " << gain << " (" << cut_value << " -> "
            << new_cut_value << ")";

        if (gain > 0) {
          cut_value = new_cut_value;
          activate_block(block1);
          activate_block(block2);
        } else {
          revert_moves(moves);
        }
      }

      const double relative_improvement =
          (prev_cut_value - cut_value) / static_cast<double>(prev_cut_value);

      DBG << "Finished round with a relative improvement of " << relative_improvement;
      if (num_round == _f_ctx.max_num_rounds ||
          relative_improvement < _f_ctx.min_round_improvement_factor) {
        break;
      }

      std::fill(_active_blocks.begin(), _active_blocks.end(), false);
      std::swap(_block_pairs, _next_block_pairs);

      found_improvement = true;
      prev_cut_value = cut_value;
    }

    return found_improvement;
  }

private:
  void apply_moves(const std::vector<Move> &moves) {
    for (const Move &move : moves) {
      _p_graph->set_block(move.node, move.new_block);
    }
  }

  void revert_moves(const std::vector<Move> &moves) {
    for (const Move &move : moves) {
      _p_graph->set_block(move.node, move.old_block);
    }
  }

  void activate_block(const BlockID block) {
    const bool was_active = _active_blocks[block];
    _active_blocks[block] = true;

    if (was_active) {
      return;
    }

    const BlockID k = _p_graph->k();
    for (BlockID other_block = 0; other_block < k; ++other_block) {
      if (block == other_block || _active_blocks[other_block]) {
        continue;
      }

      _next_block_pairs.emplace(block, other_block);
    }
  }

private:
  const PartitionContext &_p_ctx;
  const TwowayFlowRefinementContext &_f_ctx;

  PartitionedGraph *_p_graph;
  const CSRGraph *_graph;

  StaticArray<bool> _active_blocks;
  std::queue<std::pair<BlockID, BlockID>> _block_pairs;
  std::queue<std::pair<BlockID, BlockID>> _next_block_pairs;
};

TwowayFlowRefiner::TwowayFlowRefiner(const Context &ctx) : _ctx(ctx) {}

TwowayFlowRefiner::~TwowayFlowRefiner() = default;

std::string TwowayFlowRefiner::name() const {
  return "Two-Way Flow Refinement";
}

void TwowayFlowRefiner::initialize([[maybe_unused]] const PartitionedGraph &p_graph) {}

bool TwowayFlowRefiner::refine(
    PartitionedGraph &p_graph, [[maybe_unused]] const PartitionContext &p_ctx
) {
  return reified(
      p_graph,
      [&](const auto &csr_graph) {
        SCOPED_TIMER("Two-Way Flow Refinement");
        SequentialBlockPairScheduler scheduler(_ctx.partition, _ctx.refinement.twoway_flow);
        return scheduler.refine(p_graph, csr_graph);
      },
      [&]([[maybe_unused]] const auto &compressed_graph) {
        LOG_WARNING << "Cannot refine a compressed graph using the two-way flow refiner.";
        return false;
      }
  );
}

} // namespace kaminpar::shm
