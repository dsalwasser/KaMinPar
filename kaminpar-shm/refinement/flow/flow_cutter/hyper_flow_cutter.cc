#include "kaminpar-shm/refinement/flow/flow_cutter/hyper_flow_cutter.h"

#ifdef KAMINPAR_WHFC_FOUND

#include "kaminpar-shm/refinement/flow/rebalancer/dynamic_flow_rebalancer.h"
#include "kaminpar-shm/refinement/flow/rebalancer/static_flow_rebalancer.h"

#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

HyperFlowCutter::HyperFlowCutter(
    const PartitionContext &p_ctx,
    const FlowCutterContext &fc_ctx,
    const PartitionedCSRGraph &p_graph,
    GainCache &gain_cache
)
    : _p_ctx(p_ctx),
      _fc_ctx(fc_ctx),
      _sequential_flow_cutter(_hypergraph, Random::get_seed(), fc_ctx.piercing.deterministic),
      _parallel_flow_cutter(_hypergraph, Random::get_seed(), fc_ctx.piercing.deterministic),
      _delta_p_graph(&p_graph),
      _delta_gain_cache(gain_cache, _delta_p_graph) {
  _sequential_flow_cutter.timer.active = false;
  _sequential_flow_cutter.find_most_balanced = false;
  _sequential_flow_cutter.forceSequential(true);
  _sequential_flow_cutter.setBulkPiercing(fc_ctx.piercing.bulk_piercing);

  _parallel_flow_cutter.timer.active = false;
  _parallel_flow_cutter.find_most_balanced = false;
  _parallel_flow_cutter.forceSequential(false);
  _parallel_flow_cutter.setBulkPiercing(fc_ctx.piercing.bulk_piercing);

  if (_fc_ctx.rebalancer.enabled) {
    if (_fc_ctx.rebalancer.dynamic_rebalancer) {
      _flow_rebalancer = std::make_unique<
          DynamicFlowRebalancer<CSRGraph, DeltaPartitionedCSRGraph, DeltaGainCache>>(
          _delta_p_graph, _delta_gain_cache, p_ctx.max_block_weights()
      );
    } else {
      _flow_rebalancer = std::make_unique<
          StaticFlowRebalancer<CSRGraph, DeltaPartitionedCSRGraph, DeltaGainCache>>(
          _delta_p_graph, _delta_gain_cache, p_ctx.max_block_weights()
      );
    }
  }
}

HyperFlowCutter::Result HyperFlowCutter::compute_cut(
    const BorderRegion &border_region, const FlowNetwork &flow_network, bool run_sequentially
) {
  SCOPED_TIMER("Run WHFC");

  run_sequentially = run_sequentially || (flow_network.graph.n() + flow_network.graph.m()) <
                                             _fc_ctx.small_flow_network_threshold;

  initialize(flow_network);
  if (run_sequentially) {
    run_flow_cutter(_sequential_flow_cutter, border_region, flow_network);
  } else {
    run_flow_cutter(_parallel_flow_cutter, border_region, flow_network);
  }

  if (time_limit_exceeded()) {
    return Result::time_limit();
  }

  return Result(_gain, _improve_balance, _moves);
}

void HyperFlowCutter::initialize(const FlowNetwork &flow_network) {
  TIMED_SCOPE("Construct Hypergraph") {
    _hypergraph.reinitialize(flow_network.graph.n());

    const CSRGraph &graph = flow_network.graph;
    for (const NodeID u : graph.nodes()) {
      _hypergraph.nodeWeight(whfc::Node(u)) = whfc::NodeWeight(graph.node_weight(u));

      graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight c) {
        if (u < v) {
          _hypergraph.startHyperedge(whfc::Flow(c));
          _hypergraph.addPin(whfc::Node(u));
          _hypergraph.addPin(whfc::Node(v));
        }
      });
    }

    _hypergraph.finalize();
  };

  _gain = 0;
  _improve_balance = false;
  _moves.clear();
}

template <typename FlowCutter>
void HyperFlowCutter::run_flow_cutter(
    FlowCutter &flow_cutter, const BorderRegion &border_region, const FlowNetwork &flow_network
) {
  SCOPED_TIMER("Run HyperFlowCutter");

  const NodeWeight max_source_side_weight = _p_ctx.max_block_weight(border_region.block1());
  const NodeWeight max_sink_side_weight = _p_ctx.max_block_weight(border_region.block2());
  const NodeWeight total_weight = flow_network.block1_weight + flow_network.block2_weight;

  DBG << "Starting refinement for block pair " << border_region.block1() << " and "
      << border_region.block2() << " with an initial cut of " << flow_network.cut_value;

  if (_fc_ctx.rebalancer.enabled) {
    _delta_p_graph.clear();
    _delta_gain_cache.clear();
    _flow_rebalancer->initialize(border_region, flow_network);
  }

  bool found_improved_cut = false;
  EdgeWeight prev_rebalancing_gain = kInvalidEdgeWeight;

  auto &cutter_state = flow_cutter.cs;
  const auto on_cut = [&] {
    const EdgeWeight cut_value = cutter_state.flow_algo.flow_value;
    DBG << "Found a cut for block pair " << border_region.block1() << " and "
        << border_region.block2() << " with value " << cut_value;

    if (cutter_state.isBalanced()) {
      DBG << "Found cut for block pair " << border_region.block1() << " and "
          << border_region.block2() << " is a balanced cut";
      return true;
    }

    const EdgeWeight source_side_weight = cutter_state.source_reachable_weight;
    const EdgeWeight sink_side_weight = cutter_state.target_reachable_weight;

    if (_fc_ctx.rebalancer.enabled) {
      const bool rebalance_source_side_for_source_side_cut =
          source_side_weight >= (total_weight - source_side_weight);

      const bool rebalance_source_side_for_sink_side_cut =
          (total_weight - sink_side_weight) >= sink_side_weight;

      if (_fc_ctx.rebalancer.rebalance_both_cuts) {
        rebalance(
            kSourceTag,
            rebalance_source_side_for_source_side_cut,
            cut_value,
            cutter_state,
            border_region,
            flow_network
        );

        rebalance(
            kSinkTag,
            rebalance_source_side_for_sink_side_cut,
            cut_value,
            cutter_state,
            border_region,
            flow_network
        );
      } else {
        const EdgeWeight source_side_cut_overload =
            rebalance_source_side_for_source_side_cut
                ? (source_side_weight - max_source_side_weight)
                : ((total_weight - source_side_weight) - max_sink_side_weight);

        const EdgeWeight sink_side_cut_overload =
            rebalance_source_side_for_sink_side_cut
                ? ((total_weight - sink_side_weight) - max_source_side_weight)
                : (sink_side_weight - max_sink_side_weight);

        const bool rebalance_source_side_cut = source_side_cut_overload <= sink_side_cut_overload;
        const bool rebalance_source_side = rebalance_source_side_cut
                                               ? rebalance_source_side_for_source_side_cut
                                               : rebalance_source_side_for_sink_side_cut;

        rebalance(
            rebalance_source_side_cut,
            rebalance_source_side,
            cut_value,
            cutter_state,
            border_region,
            flow_network
        );
      }

      const EdgeWeight flow_cutter_gain = flow_network.cut_value - cut_value;
      if (_fc_ctx.rebalancer.abort_on_candidate_cut && _gain > flow_cutter_gain) {
        return false;
      }

      if (_fc_ctx.rebalancer.abort_on_improved_cut && _gain > 0) {
        return false;
      }

      if (_fc_ctx.rebalancer.abort_on_stable_improved_cut) {
        if (found_improved_cut) {
          const double relative_rebalancing_improvement =
              (prev_rebalancing_gain - _gain) / static_cast<double>(prev_rebalancing_gain);

          if (relative_rebalancing_improvement < 0) {
            return false;
          }
        }

        found_improved_cut |= _gain > 0;
        prev_rebalancing_gain = _gain;
      }
    }

    if (_fc_ctx.abort_on_first_cut) {
      return false;
    }

    if (cutter_state.side_to_pierce == 0) {
      DBG << "Piercing on source-side (" << source_side_weight << "/" << max_source_side_weight
          << ", " << (total_weight - source_side_weight) << "/" << max_sink_side_weight << ")";
    } else {
      DBG << "Piercing on sink-side (" << sink_side_weight << "/" << max_sink_side_weight << ", "
          << (total_weight - sink_side_weight) << "/" << max_source_side_weight << ")";
    }

    if (time_limit_exceeded()) {
      return false;
    }

    return true;
  };

  flow_cutter.cs.setMaxBlockWeight(0, std::max(flow_network.block1_weight, max_source_side_weight));
  flow_cutter.cs.setMaxBlockWeight(1, std::max(flow_network.block2_weight, max_sink_side_weight));

  flow_cutter.reset();
  flow_cutter.setFlowBound(flow_network.cut_value);

  if (_fc_ctx.piercing.determine_distance_from_cut) {
    compute_distances(border_region, flow_network, flow_cutter.cs.border_nodes.distance);
    flow_cutter.cs.border_nodes.updateMaxDistance();
  }

  const bool success = flow_cutter.enumerateCutsUntilBalancedOrFlowBoundExceeded(
      whfc::Node(flow_network.source), whfc::Node(flow_network.sink), on_cut
  );

  const EdgeWeight cut_value = cutter_state.flow_algo.flow_value;
  DBG << "Found a cut for block pair " << border_region.block1() << " and "
      << border_region.block2() << " with value " << cut_value;

  if (success) {
    const EdgeWeight flow_cutter_gain = flow_network.cut_value - cut_value;

    if (flow_cutter_gain > _gain) {
      _gain = flow_cutter_gain;
      _improve_balance =
          std::max<NodeWeight>(cutter_state.source_weight, cutter_state.target_weight) <
          std::max(flow_network.block1_weight, flow_network.block2_weight);

      compute_moves(border_region, flow_network, cutter_state);
    }
  } else if (cut_value > flow_network.cut_value) {
    DBG << "Cut is worse than the initial cut (" << flow_network.cut_value << "); "
        << "aborting refinement for block pair " << border_region.block1() << " and "
        << border_region.block2();
  }
}

void HyperFlowCutter::compute_distances(
    const BorderRegion &border_region,
    const FlowNetwork &flow_network,
    std::vector<whfc::HopDistance> &distances
) {
  whfc::HopDistance max_dist_source(0);
  whfc::HopDistance max_dist_sink(0);

  const NodeID source = flow_network.source;
  const NodeID sink = flow_network.sink;
  const CSRGraph &graph = flow_network.graph;

  distances.assign(graph.n(), whfc::HopDistance(0));

  _bfs_runner.reset();
  _bfs_marker.reset();
  _bfs_marker.resize(graph.n());

  for (const NodeID u : border_region.initial_nodes_region1()) {
    const NodeID u_local = flow_network.global_to_local_mapping.get(u);
    _bfs_marker.set(u_local);
    _bfs_runner.add_seed(u_local);
  }
  for (const NodeID u : border_region.initial_nodes_region2()) {
    const NodeID u_local = flow_network.global_to_local_mapping.get(u);
    _bfs_marker.set(u_local);
    _bfs_runner.add_seed(u_local);
  }

  _bfs_runner.perform(1, [&](const NodeID u, const NodeID u_distance, auto &queue) {
    const NodeID u_global = flow_network.local_to_global_mapping.get(u);
    const bool source_side = border_region.region1_contains(u_global);

    const whfc::HopDistance dist(u_distance);
    if (source_side) {
      distances[u] = -dist;
      max_dist_source = std::max(max_dist_source, dist);
    } else {
      distances[u] = dist;
      max_dist_sink = std::max(max_dist_sink, dist);
    }

    graph.adjacent_nodes(u, [&](const NodeID v) {
      if (v == source || v == sink || _bfs_marker.get(v)) {
        return;
      }

      _bfs_marker.set(v);
      queue.push_back(v);
    });
  });

  distances[source] = -(max_dist_source + 1);
  distances[sink] = max_dist_sink + 1;
}

template <typename CutterState>
void HyperFlowCutter::compute_moves(
    const BorderRegion &border_region,
    const FlowNetwork &flow_network,
    const CutterState &cutter_state
) {
  const BlockID block1 = border_region.block1();
  const BlockID block2 = border_region.block2();
  const auto &flow_algorithm = cutter_state.flow_algo;

  _moves.clear();
  for (const auto &[u, u_local] : flow_network.global_to_local_mapping.entries()) {
    const BlockID old_block = border_region.region1_contains(u) ? block1 : block2;
    const BlockID new_block = flow_algorithm.isSource(whfc::Node(u_local)) ? block1 : block2;

    if (old_block != new_block) {
      _moves.emplace_back(u, old_block, new_block);
    }
  }
}

template <typename CutterState>
void HyperFlowCutter::rebalance(
    const bool source_side_cut,
    const bool rebalance_source_side,
    const EdgeWeight cur_cut_value,
    const CutterState &cutter_state,
    const BorderRegion &border_region,
    const FlowNetwork &flow_network
) {
  SCOPED_TIMER("Rebalance");

  const BlockID block1 = border_region.block1();
  const BlockID block2 = border_region.block2();

  const BlockID side = source_side_cut ? block1 : block2;
  const BlockID other_side = source_side_cut ? block2 : block1;

  TIMED_SCOPE("Update State") {
    for (const auto &[u, u_local] : flow_network.global_to_local_mapping.entries()) {
      const BlockID old_block = _delta_p_graph.block(u);
      const BlockID new_block =
          (source_side_cut ? cutter_state.flow_algo.isSourceReachable(whfc::Node(u_local))
                           : cutter_state.flow_algo.isTargetReachable(whfc::Node(u_local)))
              ? side
              : other_side;

      if (old_block != new_block) {
        _delta_p_graph.set_block(u, new_block);
        _delta_gain_cache.move(u, old_block, new_block);
      }
    }
  };

  const BlockID overloaded_block = rebalance_source_side ? block1 : block2;
  const RebalanceResult rebalancer_result = _flow_rebalancer->rebalance(overloaded_block);

  if (!rebalancer_result.balanced) {
    DBG << "Rebalancer failed to produce a balanced cut";
  } else {
    const EdgeWeight flow_cutter_gain = flow_network.cut_value - cur_cut_value;
    const EdgeWeight total_gain = flow_cutter_gain + rebalancer_result.gain;
    DBG << "Rebalanced imbalanced " << (source_side_cut ? "source-side" : "sink-side")
        << " cut with gain " << total_gain;

    if (total_gain > _gain) {
      SCOPED_TIMER("Compute Moves");

      _gain = total_gain;
      _moves.clear();

      for (const NodeID u : border_region.nodes_region1()) {
        const BlockID new_block = _delta_p_graph.block(u);

        if (new_block != block1) {
          _moves.emplace_back(u, block1, new_block);
        };
      }

      for (const NodeID u : border_region.nodes_region2()) {
        const BlockID new_block = _delta_p_graph.block(u);

        if (new_block != block2) {
          _moves.emplace_back(u, block2, new_block);
        };
      }

      for (const auto &[u, target_block] : rebalancer_result.moves) {
        if (flow_network.global_to_local_mapping.contains(u)) {
          continue;
        }

        _moves.emplace_back(u, overloaded_block, target_block);
      }
    }
  }

  _flow_rebalancer->revert_moves();
}

void HyperFlowCutter::free() {
  _bfs_marker.free();
  _bfs_runner.free();

  _moves.clear();
  _moves.shrink_to_fit();
}

} // namespace kaminpar::shm

#endif
