#include "kaminpar-shm/refinement/flow/scheduler/sequential_active_block_scheduler.h"

#include <algorithm>

#include "kaminpar-shm/metrics.h"
#include "kaminpar-shm/refinement/flow/rebalancer/round_static_flow_rebalancer.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

SequentialActiveBlockScheduler::SequentialActiveBlockScheduler(
    const TwowayFlowRefinementContext &f_ctx
)
    : _f_ctx(f_ctx),
      _active_block_scheduling(f_ctx.scheduler) {}

bool SequentialActiveBlockScheduler::refine(
    PartitionedCSRGraph &p_graph, const CSRGraph &graph, const PartitionContext &p_ctx
) {
  _p_graph = &p_graph;
  _graph = &graph;

  if (_active_blocks.size() < p_graph.k()) {
    _active_blocks.resize(p_graph.k(), static_array::noinit);
  }
  std::fill_n(_active_blocks.begin(), p_graph.k(), true);

  // Since the timers have a significant running time overhead, we disable them usually.
  IF_NOT_DBG DISABLE_TIMERS();
  IF_STATS _stats.reset();

  const TimePoint start_time = Clock::now();
  QuotientGraph quotient_graph(p_graph);

  if (_f_ctx.flow_cutter.rebalancer.enabled) {
    if (_f_ctx.flow_cutter.use_whfc) {
      LOG_WARNING << "Cannot use the flow rebalancer together with WHFC; disabling rebalancing.";
    }

    _gain_cache.initialize(graph, p_graph);
  }

  FlowRefiner refiner(p_ctx, _f_ctx, quotient_graph, p_graph, graph, _gain_cache, start_time);

  std::size_t num_round = 0;
  bool found_improvement = false;

  double prev_imbalance = metrics::imbalance(p_graph);
  EdgeWeight prev_cut_value = quotient_graph.total_cut_weight();

  while (prev_cut_value > 0) {
    num_round += 1;
    DBG << "Starting round " << num_round;

    if (_f_ctx.flow_cutter.rebalancer.enabled) {
      initialize_rebalancers(p_graph, graph, p_ctx);
    }

    const SubroundScheduling active_block_pairs = TIMED_SCOPE("Compute Active Block Pairs") {
      return _active_block_scheduling.compute_subround_scheduling(
          quotient_graph, _active_blocks, num_round
      );
    };
    std::fill_n(_active_blocks.begin(), p_graph.k(), false);

    EdgeWeight cut_value = prev_cut_value;
    for (const auto &[block1, block2] : active_block_pairs) {
      IF_STATS _stats.num_searches += 1;
      DBG << "Scheduling block pair " << block1 << " and " << block2;

      const Result result = refiner.refine(
          block1, block2, flow_rebalancer_moves(block1, block2), _f_ctx.run_sequentially
      );

      if (result.time_limit_exceeded) {
        LOG_WARNING << "Time limit exceeded during flow refinement";
        num_round = _f_ctx.max_num_rounds;
        break;
      }

      const EdgeWeight new_cut_value = cut_value - result.gain;
      DBG << "Found balanced cut for block pair " << block1 << " and " << block2 << " with gain "
          << result.gain << " (" << cut_value << " -> " << new_cut_value << ")";

      if (result.gain > 0 || (result.gain == 0 && result.improved_balance)) {
        apply_moves(result.moves);

        KASSERT(
            metrics::is_balanced(p_graph, p_ctx),
            "Computed an imbalanced move sequence",
            assert::heavy
        );
        KASSERT(
            metrics::edge_cut_seq(p_graph) == new_cut_value,
            "Computed an invalid new cut value",
            assert::heavy
        );

        IF_STATS _stats.num_improvements += 1;

        cut_value = new_cut_value;

        quotient_graph.add_cut_edges(_new_cut_edges);
        quotient_graph.add_gain(block1, block2, result.gain);

        _active_blocks[block1] = true;
        _active_blocks[block2] = true;
      }
    }

    const EdgeWeight round_gain = prev_cut_value - cut_value;
    const double imbalance = metrics::imbalance(p_graph);
    found_improvement |= round_gain > 0 || imbalance < prev_imbalance;

    const double relative_improvement = round_gain / static_cast<double>(prev_cut_value);
    DBG << "Finished round with a relative improvement of " << relative_improvement
        << " and imbalance of " << imbalance;

    if (num_round == _f_ctx.max_num_rounds ||
        relative_improvement < _f_ctx.min_round_improvement_factor) {
      break;
    }

    quotient_graph.reconstruct();

    prev_cut_value = cut_value;
    prev_imbalance = imbalance;
  }

  IF_NOT_DBG ENABLE_TIMERS();
  IF_STATS _stats.print();

  return found_improvement;
}

void SequentialActiveBlockScheduler::initialize_rebalancers(
    const PartitionedCSRGraph &p_graph, const CSRGraph &graph, const PartitionContext &p_ctx
) {
  if (_f_ctx.flow_cutter.rebalancer.kind != FlowRebalancerKind::ROUND_STATIC) {
    return;
  }

  if (_nodes_per_block.size() < p_graph.k()) {
    _nodes_per_block.resize(p_graph.k());
  }
  for (auto &nodes : _nodes_per_block) {
    nodes.clear();
  }

  if (_moves_per_block.size() < p_graph.k()) {
    _moves_per_block.resize(p_graph.k());
  }

  for (const NodeID u : graph.nodes()) {
    const BlockID u_block = p_graph.block(u);
    _nodes_per_block[u_block].push_back(u);
  }

  RoundStaticFlowRebalancer<GainCache> flow_rebalancer(
      p_graph, _gain_cache, p_ctx.max_block_weights()
  );
  for (BlockID block = 0; block < p_graph.k(); ++block) {
    _moves_per_block[block] = flow_rebalancer.compute_moves(block, _nodes_per_block[block]);
  }
}

FlowRebalancerMoves
SequentialActiveBlockScheduler::flow_rebalancer_moves(const BlockID block1, const BlockID block2) {
  return (_f_ctx.flow_cutter.rebalancer.enabled &&
          _f_ctx.flow_cutter.rebalancer.kind == FlowRebalancerKind::ROUND_STATIC)
             ? FlowRebalancerMoves(_moves_per_block[block1], _moves_per_block[block2])
             : FlowRebalancerMoves();
}

void SequentialActiveBlockScheduler::apply_moves(std::span<const Move> moves) {
  SCOPED_TIMER("Apply Moves");

  _new_cut_edges.clear();
  for (const Move &move : moves) {
    KASSERT(
        _p_graph->block(move.node) == move.old_block,
        "Move sequence contains invalid old block ids",
        assert::heavy
    );
    KASSERT(
        move.old_block != move.new_block,
        "Move sequence contains moves where node is already in target block",
        assert::heavy
    );

    const NodeID u = move.node;
    const BlockID new_block = move.new_block;
    _p_graph->set_block(u, new_block);

    const BlockID old_block = move.old_block;
    _graph->adjacent_nodes(u, [&](const NodeID v) {
      const BlockID v_block = _p_graph->block(v);

      if (v_block == old_block) {
        _new_cut_edges.emplace_back(u, v);
      }
    });
  }

  if (_f_ctx.flow_cutter.rebalancer.enabled) {
    for (const Move &move : moves) {
      _gain_cache.move(move.node, move.old_block, move.new_block);
    }
  }
}

} // namespace kaminpar::shm
