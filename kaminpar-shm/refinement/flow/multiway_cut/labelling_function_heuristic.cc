#include "kaminpar-shm/refinement/flow/multiway_cut/labelling_function_heuristic.h"

#include <algorithm>
#include <limits>
#include <utility>

#include "kaminpar-shm/refinement/flow/max_flow/edmond_karp_algorithm.h"
#include "kaminpar-shm/refinement/flow/max_flow/fifo_preflow_push_algorithm.h"
#include "kaminpar-shm/refinement/flow/util/reverse_edge_index.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/random.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

LabellingFunctionHeuristic::LabellingFunctionHeuristic(const LabellingFunctionHeuristicContext &ctx)
    : _ctx(ctx) {
  switch (FlowAlgorithm::FIFO_PREFLOW_PUSH) {
  case FlowAlgorithm::EDMONDS_KARP:
    _max_flow_algorithm = std::make_unique<EdmondsKarpAlgorithm>();
    break;
  case FlowAlgorithm::FIFO_PREFLOW_PUSH:
    _max_flow_algorithm = std::make_unique<FIFOPreflowPushAlgorithm>(ctx.fifo_preflow_push);
    break;
  }
}

MultiwayCutAlgorithm::Result LabellingFunctionHeuristic::compute(
    const PartitionedCSRGraph &p_graph,
    const CSRGraph &graph,
    [[maybe_unused]] std::span<const NodeID> reverse_edges,
    const TerminalSets &terminal_sets
) {
  _p_graph = &p_graph;
  _graph = &graph;
  _terminal_sets = &terminal_sets;

  initialize_labelling_function();
  improve_labelling_function();
  std::unordered_set<EdgeID> cut_edges = derive_cut_edges();

  KASSERT(
      debug::is_valid_multiway_cut(*_graph, terminal_sets, cut_edges),
      "computed a non-valid multi-way cut using the labelling-function heuristic",
      assert::heavy
  );

  return Result(_labelling_function_cost, std::move(cut_edges));
}

void LabellingFunctionHeuristic::initialize_labelling_function() {
  SCOPED_TIMER("Initialize Labelling Function");

  if (_labelling_function.size() < _graph->n()) {
    _labelling_function.resize(_graph->n(), static_array::noinit);
  }

  switch (_ctx.initialization_strategy) {
  case LabellingFunctionInitializationStrategy::ZERO: {
    std::fill_n(_labelling_function.begin(), _graph->n(), 0);
    break;
  }
  case LabellingFunctionInitializationStrategy::RANDOM: {
    const BlockID num_terminal_sets = _terminal_sets->num_terminal_sets();
    Random &random = Random::instance();

    for (const NodeID u : _graph->nodes()) {
      _labelling_function[u] = random.random_index(0, num_terminal_sets);
    }

    break;
  }
  case LabellingFunctionInitializationStrategy::EXISTING_PARTITION: {
    for (const NodeID u : _graph->nodes()) {
      _labelling_function[u] = _p_graph->block(u);
    }

    break;
  }
  }

  BlockID label = 0;
  for (BlockID terminal_set = 0; terminal_set < _terminal_sets->num_terminal_sets();
       ++terminal_set) {
    const BlockID terminal_label = label++;

    for (const NodeID u : _terminal_sets->terminal_set_nodes(terminal_set)) {
      _labelling_function[u] = terminal_label;
    }
  }
}

void LabellingFunctionHeuristic::improve_labelling_function() {
  KASSERT(is_valid_labelling_function(), "invalid labelling function");

  EdgeWeight cur_cost = compute_labelling_function_cost();
  DBG << "Starting labelling function local search with a initial cost " << cur_cost;

  const BlockID num_terminal_sets = _terminal_sets->num_terminal_sets();
  while (true) {
    bool found_improvement = false;

    for (BlockID terminal = 0; terminal < num_terminal_sets; ++terminal) {
      FlowNetwork flow_network = construct_flow_network(terminal);

      DBG << "Constructed a flow network with n=" << flow_network.graph.n()
          << " and m=" << flow_network.graph.m();

      TIMED_SCOPE("Initialize Max Flow Algorithm") {
        _max_flow_algorithm->initialize(
            flow_network.graph, flow_network.reverse_edges, flow_network.source, flow_network.sink
        );
      };

      const auto [cost, flow] = TIMED_SCOPE("Compute Max Flow") {
        return _max_flow_algorithm->compute_max_flow();
      };
      DBG << "Computed a labelling function with cost " << cost;

      if (cost < (1 - _ctx.epsilon) * cur_cost) {
        found_improvement = true;
        cur_cost = cost;

        derive_labelling_function(terminal, flow_network, flow);

        KASSERT(
            is_valid_labelling_function(),
            "computed an invalid labelling function update",
            assert::heavy
        );
        KASSERT(
            cost == compute_labelling_function_cost(),
            "computed an invalid labelling function cost",
            assert::heavy
        );
      }
    }

    if (!found_improvement) {
      break;
    }
  }

  _labelling_function_cost = cur_cost;
}

EdgeWeight LabellingFunctionHeuristic::compute_labelling_function_cost() const {
  SCOPED_TIMER("Compute Labelling Function Cost");

  EdgeWeight cost = 0;

  for (const NodeID u : _graph->nodes()) {
    _graph->adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
      cost += (_labelling_function[u] != _labelling_function[v]) ? w : 0;
    });
  }

  return cost / 2;
}

LabellingFunctionHeuristic::FlowNetwork
LabellingFunctionHeuristic::construct_flow_network(const BlockID label) {
  SCOPED_TIMER("Construct Flow Network");

  constexpr NodeID kSource = 0;
  constexpr NodeID kSink = 1;
  constexpr EdgeWeight kInfinity = std::numeric_limits<EdgeWeight>::max();

  const BlockID num_terminal_sets = _terminal_sets->num_terminal_sets();
  const NodeID num_terminals = _terminal_sets->num_terminals();

  const NodeID kFirstTerminalNodeID = 2;
  const NodeID kFirstRealNodeID = kFirstTerminalNodeID + num_terminal_sets;
  const NodeID kFirstEdgeNodeID = kFirstRealNodeID + (_graph->n() - num_terminals);

  if (_edge_collector.size() < _graph->n() + num_terminal_sets) {
    _edge_collector.resize(_graph->n() + num_terminal_sets);
  }

  EdgeID num_terminal_edge_nodes = 0;
  for (BlockID terminal_set = 0; terminal_set < num_terminal_sets; ++terminal_set) {
    for (const NodeID u : _terminal_sets->terminal_set_nodes(terminal_set)) {
      _graph->adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
        if (_terminal_sets->is_terminal(v)) {
          const BlockID v_terminal_set = _terminal_sets->terminal_set(v);

          if (terminal_set < v_terminal_set) {
            _edge_collector[_graph->n() + v_terminal_set] += w;
          }
        } else {
          _edge_collector[v] += w;
        }
      });
    }

    Neighborhood terminal_tneighborhood;
    Neighborhood terminal_neighborhood;
    for (const auto [v, w] : _edge_collector.entries()) {
      if (v >= _graph->n()) {
        num_terminal_edge_nodes += 1;

        const NodeID other_terminal_set = v - _graph->n();
        terminal_tneighborhood.emplace_back(other_terminal_set, w);
      } else {
        const BlockID v_label = _labelling_function[v];
        if (terminal_set != v_label) {
          num_terminal_edge_nodes += 1;
        }

        terminal_neighborhood.emplace_back(v, w);
      }
    }

    _terminal_tneighborhoods.push_back(std::move(terminal_tneighborhood));
    _terminal_neighborhoods.push_back(std::move(terminal_neighborhood));

    _edge_collector.clear();
  }

  StaticArray<NodeID> remapping(_graph->n(), static_array::noinit);

  NodeID num_nodes =
      2 + num_terminal_sets + (_graph->n() - num_terminals) + num_terminal_edge_nodes;
  for (const NodeID u : _graph->nodes()) {
    const bool is_terminal = _terminal_sets->is_terminal(u);
    remapping[u] = is_terminal ? 0 : 1;

    if (is_terminal) {
      continue;
    }

    const BlockID u_label = _labelling_function[u];
    _graph->adjacent_nodes(u, [&](const NodeID v) {
      if (u >= v || _terminal_sets->is_terminal(v)) {
        return;
      }

      const BlockID v_label = _labelling_function[v];
      num_nodes += (u_label != v_label) ? 1 : 0;
    });
  }

  std::partial_sum(remapping.begin(), remapping.end(), remapping.begin());

  StaticArray<EdgeID> nodes(num_nodes + 1, static_array::noinit);
  std::fill_n(nodes.begin(), nodes.size(), 0);

  NodeID cur_edge = kFirstEdgeNodeID;
  for (BlockID terminal_set = 0; terminal_set < num_terminal_sets; ++terminal_set) {
    const BlockID terminal_label = terminal_set;

    nodes[kFirstTerminalNodeID + terminal_set] += 1;
    if (terminal_set == label) {
      nodes[kSink] += 1;
    } else {
      nodes[kSource] += 1;
    }

    for (const auto [other_terminal_set, w] : _terminal_tneighborhoods[terminal_set]) {
      KASSERT(terminal_set < other_terminal_set);
      const BlockID other_terminal_label = other_terminal_set;

      nodes[kSink] += 1;
      nodes[cur_edge] += 1;

      if (terminal_label != label) {
        nodes[kFirstTerminalNodeID + terminal_set] += 1;
        nodes[cur_edge] += 1;
      }

      if (other_terminal_label != label) {
        nodes[kFirstTerminalNodeID + other_terminal_set] += 1;
        nodes[cur_edge] += 1;
      }

      cur_edge += 1;
    }

    for (const auto [v, w] : _terminal_neighborhoods[terminal_set]) {
      const NodeID v_remapped = remapping[v] - 1;
      const BlockID v_label = _labelling_function[v];

      if (terminal_label == v_label) {
        nodes[kFirstTerminalNodeID + terminal_set] += 1;
        nodes[kFirstRealNodeID + v_remapped] += 1;
      } else {
        nodes[kSink] += 1;
        nodes[cur_edge] += 1;

        if (terminal_label != label) {
          nodes[kFirstTerminalNodeID + terminal_set] += 1;
          nodes[cur_edge] += 1;
        }

        if (v_label != label) {
          nodes[kFirstRealNodeID + v_remapped] += 1;
          nodes[cur_edge] += 1;
        }

        cur_edge += 1;
      }
    }
  }

  for (const NodeID u : _graph->nodes()) {
    if (_terminal_sets->is_terminal(u)) {
      continue;
    }

    const NodeID u_remapped = remapping[u] - 1;
    const BlockID u_label = _labelling_function[u];
    if (u_label == label) {
      nodes[kSink] += 1;
      nodes[kFirstRealNodeID + u_remapped] += 1;
    }

    _graph->adjacent_nodes(u, [&](const NodeID v) {
      if (u >= v || _terminal_sets->is_terminal(v)) {
        return;
      }

      const NodeID v_remapped = remapping[v] - 1;
      const BlockID v_label = _labelling_function[v];
      if (u_label == v_label) {
        nodes[kFirstRealNodeID + u_remapped] += 1;
        nodes[kFirstRealNodeID + v_remapped] += 1;
      } else {
        nodes[kSink] += 1;
        nodes[cur_edge] += 1;

        if (u_label != label) {
          nodes[kFirstRealNodeID + u_remapped] += 1;
          nodes[cur_edge] += 1;
        }

        if (v_label != label) {
          nodes[kFirstRealNodeID + v_remapped] += 1;
          nodes[cur_edge] += 1;
        }

        cur_edge += 1;
      }
    });
  }

  std::partial_sum(nodes.begin(), nodes.end(), nodes.begin());

  const EdgeID num_edges = nodes.back();
  StaticArray<NodeID> edges(num_edges, static_array::noinit);
  StaticArray<EdgeWeight> edge_weights(num_edges, static_array::noinit);
  StaticArray<NodeID> reverse_edges(num_edges, static_array::noinit);

  const auto add_edge = [&](const NodeID u, const NodeID v, const EdgeWeight w) {
    const EdgeID e1 = --nodes[u];
    edges[e1] = v;
    edge_weights[e1] = w;

    const EdgeID e2 = --nodes[v];
    edges[e2] = u;
    edge_weights[e2] = w;

    reverse_edges[e1] = e2;
    reverse_edges[e2] = e1;

    KASSERT(u != kSource || v != kSink);
  };

  cur_edge = kFirstEdgeNodeID;
  for (BlockID terminal_set = 0; terminal_set < num_terminal_sets; ++terminal_set) {
    const BlockID terminal_label = terminal_set;

    if (terminal_set == label) {
      add_edge(kFirstTerminalNodeID + terminal_set, kSink, kInfinity);
    } else {
      add_edge(kFirstTerminalNodeID + terminal_set, kSource, kInfinity);
    }

    for (const auto [other_terminal_set, w] : _terminal_tneighborhoods[terminal_set]) {
      KASSERT(terminal_set < other_terminal_set);
      const BlockID other_terminal_label = other_terminal_set;

      add_edge(kSink, cur_edge, w);

      if (terminal_label != label) {
        add_edge(kFirstTerminalNodeID + terminal_set, cur_edge, w);
      }

      if (other_terminal_label != label) {
        add_edge(kFirstTerminalNodeID + other_terminal_set, cur_edge, w);
      }

      cur_edge += 1;
    }

    for (const auto [v, w] : _terminal_neighborhoods[terminal_set]) {
      const NodeID v_remapped = remapping[v] - 1;
      const BlockID v_label = _labelling_function[v];

      if (terminal_label == v_label) {
        add_edge(kFirstTerminalNodeID + terminal_set, kFirstRealNodeID + v_remapped, w);
      } else {
        add_edge(kSink, cur_edge, w);

        if (terminal_label != label) {
          add_edge(kFirstTerminalNodeID + terminal_set, cur_edge, w);
        }

        if (v_label != label) {
          add_edge(kFirstRealNodeID + v_remapped, cur_edge, w);
        }

        cur_edge += 1;
      }
    }
  }

  for (const NodeID u : _graph->nodes()) {
    if (_terminal_sets->is_terminal(u)) {
      continue;
    }

    const NodeID u_remapped = remapping[u] - 1;
    const BlockID u_label = _labelling_function[u];
    if (u_label == label) {
      add_edge(kSink, kFirstRealNodeID + u_remapped, kInfinity);
    }

    _graph->adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
      if (u >= v || _terminal_sets->is_terminal(v)) {
        return;
      }

      const NodeID v_remapped = remapping[v] - 1;
      const BlockID v_label = _labelling_function[v];
      if (u_label == v_label) {
        add_edge(kFirstRealNodeID + u_remapped, kFirstRealNodeID + v_remapped, w);
      } else {
        add_edge(kSink, cur_edge, w);

        if (u_label != label) {
          add_edge(kFirstRealNodeID + u_remapped, cur_edge, w);
        }

        if (v_label != label) {
          add_edge(kFirstRealNodeID + v_remapped, cur_edge, w);
        }

        cur_edge += 1;
      }
    });
  }

  CSRGraph graph(
      CSRGraph::seq(),
      std::move(nodes),
      std::move(edges),
      StaticArray<NodeWeight>(),
      std::move(edge_weights)
  );

  KASSERT(debug::validate_graph(graph), "constructed an invalid flow network", assert::heavy);
  KASSERT(
      debug::is_valid_reverse_edge_index(graph, reverse_edges),
      "constructed an invalid reverse edge index",
      assert::heavy
  );

  return FlowNetwork(
      kSource,
      kSink,
      std::move(graph),
      std::move(reverse_edges),
      kFirstRealNodeID,
      kFirstEdgeNodeID,
      std::move(remapping)
  );
}

/*
LabellingFunctionHeuristic::FlowNetwork
LabellingFunctionHeuristic::construct_flow_network(const BlockID label) const {
  SCOPED_TIMER("Construct Flow Network");

  constexpr NodeID kSource = 0;
  constexpr NodeID kSink = 1;
  constexpr NodeID kFirstNodeID = 2;
  constexpr EdgeWeight kInfinity = std::numeric_limits<EdgeWeight>::max();

  NodeID num_nodes = 2 + _graph->n();
  for (const NodeID u : _graph->nodes()) {
    const BlockID u_label = _labelling_function[u];

    _graph->adjacent_nodes(u, [&](const NodeID v) {
      if (u >= v) {
        return;
      }

      const BlockID v_label = _labelling_function[v];
      num_nodes += (u_label != v_label) ? 1 : 0;
    });
  }

  StaticArray<EdgeID> nodes(num_nodes + 1, static_array::noinit);
  std::fill_n(nodes.begin(), nodes.size(), 0);

  NodeID cur_edge = kFirstNodeID + _graph->n();
  for (const NodeID u : _graph->nodes()) {
    const BlockID u_label = _labelling_function[u];

    if (_terminal_sets->is_terminal(u) && u_label != label) {
      nodes[kSource] += 1;
      nodes[kFirstNodeID + u] += 1;
    }

    if (u_label == label) {
      nodes[kSink] += 1;
      nodes[kFirstNodeID + u] += 1;
    }

    _graph->adjacent_nodes(u, [&](const NodeID v) {
      if (u >= v) {
        return;
      }

      const BlockID v_label = _labelling_function[v];
      if (u_label == v_label) {
        nodes[kFirstNodeID + u] += 1;
        nodes[kFirstNodeID + v] += 1;
      } else {
        nodes[kSink] += 1;
        nodes[cur_edge] += 1;

        if (u_label != label) {
          nodes[kFirstNodeID + u] += 1;
          nodes[cur_edge] += 1;
        }

        if (v_label != label) {
          nodes[kFirstNodeID + v] += 1;
          nodes[cur_edge] += 1;
        }

        cur_edge += 1;
      }
    });
  }

  std::partial_sum(nodes.begin(), nodes.end(), nodes.begin());

  const EdgeID num_edges = nodes.back();
  StaticArray<NodeID> edges(num_edges, kInvalidNodeID, static_array::seq);
  StaticArray<EdgeWeight> edge_weights(num_edges, static_array::noinit);
  StaticArray<NodeID> reverse_edges(num_edges, static_array::noinit);

  LOG << num_nodes << ' ' << num_edges;
  const auto add_edge = [&](const NodeID u, const NodeID v, const EdgeWeight w) {
    const EdgeID e1 = --nodes[u];
    edges[e1] = v;
    edge_weights[e1] = w;

    const EdgeID e2 = --nodes[v];
    edges[e2] = u;
    edge_weights[e2] = w;

    reverse_edges[e1] = e2;
    reverse_edges[e2] = e1;

    KASSERT(u != kSource || v != kSink);
  };

  cur_edge = kFirstNodeID + _graph->n();
  for (const NodeID u : _graph->nodes()) {
    const BlockID u_label = _labelling_function[u];

    if (_terminal_sets->is_terminal(u) && u_label != label) {
      add_edge(kSource, kFirstNodeID + u, kInfinity);
    }

    if (u_label == label) {
      add_edge(kSink, kFirstNodeID + u, kInfinity);
    }

    _graph->adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
      if (u >= v) {
        return;
      }

      const BlockID v_label = _labelling_function[v];
      if (u_label == v_label) {
        add_edge(kFirstNodeID + u, kFirstNodeID + v, w);
      } else {
        add_edge(kSink, cur_edge, w);

        if (u_label != label) {
          add_edge(kFirstNodeID + u, cur_edge, w);
        }

        if (v_label != label) {
          add_edge(kFirstNodeID + v, cur_edge, w);
        }

        cur_edge += 1;
      }
    });
  }

  CSRGraph graph(
      CSRGraph::seq(),
      std::move(nodes),
      std::move(edges),
      StaticArray<NodeWeight>(),
      std::move(edge_weights)
  );

  KASSERT(debug::validate_graph(graph), "constructed an invalid flow network", assert::heavy);
  KASSERT(
      debug::is_valid_reverse_edge_index(graph, reverse_edges),
      "constructed an invalid reverse edge index",
      assert::heavy
  );

  return FlowNetwork(kSource, kSink, std::move(graph), std::move(reverse_edges), 0, 0, {});
}
*/

void LabellingFunctionHeuristic::derive_labelling_function(
    const BlockID label, const FlowNetwork &flow_network, std::span<const EdgeWeight> flow
) {
  SCOPED_TIMER("Derive Labelling Function");

  const std::unordered_set<NodeID> cut_nodes =
      compute_cut_nodes(flow_network.graph, flow_network.source, flow);
  for (const NodeID u : _graph->nodes()) {
    if (_terminal_sets->is_terminal(u)) {
      continue;
    }

    const NodeID u_local = flow_network.node_start + flow_network.remapping[u] - 1;
    // const NodeID u_local = 2 + u;
    if (!cut_nodes.contains(u_local)) {
      _labelling_function[u] = label;
    }
  }
}

std::unordered_set<NodeID> LabellingFunctionHeuristic::compute_cut_nodes(
    const CSRGraph &graph, const NodeID terminal, std::span<const EdgeWeight> flow
) {
  SCOPED_TIMER("Compute Cut Nodes");

  std::unordered_set<NodeID> cut_nodes;

  std::queue<NodeID> bfs_queue;
  cut_nodes.insert(terminal);
  bfs_queue.push(terminal);

  while (!bfs_queue.empty()) {
    const NodeID u = bfs_queue.front();
    bfs_queue.pop();

    graph.neighbors(u, [&](const EdgeID e, const NodeID v, const EdgeWeight c) {
      if (cut_nodes.contains(v)) {
        return;
      }

      const EdgeWeight e_flow = flow[e];
      const bool has_residual_capacity = e_flow < c;
      if (has_residual_capacity) {
        cut_nodes.insert(v);
        bfs_queue.push(v);
      }
    });
  }

  return cut_nodes;
}

std::unordered_set<EdgeID> LabellingFunctionHeuristic::derive_cut_edges() const {
  SCOPED_TIMER("Derive Cut Edges");

  std::unordered_set<EdgeID> cut_edges;

  for (const NodeID u : _graph->nodes()) {
    _graph->neighbors(u, [&](const EdgeID e, const NodeID v) {
      if (_labelling_function[u] != _labelling_function[v]) {
        cut_edges.insert(e);
      }
    });
  }

  return cut_edges;
}

bool LabellingFunctionHeuristic::is_valid_labelling_function() const {
  bool is_valid = true;

  for (BlockID terminal_set = 0; terminal_set < _terminal_sets->num_terminal_sets();
       ++terminal_set) {
    const BlockID terminal_label = terminal_set;

    for (const NodeID u : _terminal_sets->terminal_set_nodes(terminal_set)) {
      const BlockID u_label = _labelling_function[u];

      if (terminal_label != u_label) {
        LOG_WARNING << "Terminal " << u << " is assigned " << u_label << " != " << terminal_label;
        is_valid = false;
      }
    }
  }

  return is_valid;
}

} // namespace kaminpar::shm