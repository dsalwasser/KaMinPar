/*******************************************************************************
 * @file:   kway.cc
 * @author: Daniel Seemaier
 * @date:   25.10.2021
 * @brief:  Partitioning scheme using direct k-way partitioning.
 ******************************************************************************/
#include "dkaminpar/partitioning_scheme/kway_partitioning_scheme.h"

#include "dkaminpar/coarsening/coarsener.h"
#include "dkaminpar/coarsening/global_clustering_contraction.h"
#include "dkaminpar/datastructure/distributed_graph.h"
#include "dkaminpar/debug.h"
#include "dkaminpar/factories.h"
#include "dkaminpar/graphutils/allgather_graph.h"
#include "dkaminpar/io.h"
#include "dkaminpar/metrics.h"
#include "dkaminpar/refinement/distributed_balancer.h"

#include "kaminpar/metrics.h"

#include "common/console_io.h"
#include "common/timer.h"
#include "common/utils/strings.h"

namespace kaminpar::dist {
SET_DEBUG(false);

KWayPartitioningScheme::KWayPartitioningScheme(const DistributedGraph& graph, const Context& ctx)
    : _graph{graph},
      _ctx{ctx} {}

DistributedPartitionedGraph KWayPartitioningScheme::partition() {
    Coarsener coarsener(_graph, _ctx);

    const DistributedGraph* graph = &_graph;

    ////////////////////////////////////////////////////////////////////////////////
    // Step 1: Coarsening
    ////////////////////////////////////////////////////////////////////////////////
    if (mpi::get_comm_rank(_graph.communicator()) == 0) {
        cio::print_banner("Coarsening");
    }

    {
        SCOPED_TIMER("Coarsening");

        while (graph->global_n() > _ctx.parallel.num_threads * _ctx.partition.k * _ctx.coarsening.contraction_limit) {
            SCOPED_TIMER("Coarsening", std::string("Level ") + std::to_string(coarsener.level()));
            const GlobalNodeWeight max_cluster_weight = coarsener.max_cluster_weight();

            const DistributedGraph* c_graph   = coarsener.coarsen_once();
            const bool              converged = (graph == c_graph);

            if (!converged) {
                if (_ctx.debug.save_graph_hierarchy) {
                    debug::save_graph(*c_graph, _ctx, static_cast<int>(coarsener.level()));
                }

                // Print statistics for coarse graph
                const std::string n_str       = mpi::gather_statistics_str(c_graph->n(), c_graph->communicator());
                const std::string ghost_n_str = mpi::gather_statistics_str(c_graph->ghost_n(), c_graph->communicator());
                const std::string m_str       = mpi::gather_statistics_str(c_graph->m(), c_graph->communicator());
                const std::string max_node_weight_str =
                    mpi::gather_statistics_str<GlobalNodeWeight>(c_graph->max_node_weight(), c_graph->communicator());

                // Machine readable
                LOG << "=> level=" << coarsener.level() << " "
                    << "global_n=" << c_graph->global_n() << " "
                    << "global_m=" << c_graph->global_m() << " "
                    << "n=[" << n_str << "] "
                    << "ghost_n=[" << ghost_n_str << "] "
                    << "m=[" << m_str << "] "
                    << "max_node_weight=[" << max_node_weight_str << "] "
                    << "max_cluster_weight=" << max_cluster_weight;

                // Human readable
                LOG << "Level " << coarsener.level() << ":";
                graph::print_summary(*c_graph);

                graph = c_graph;
            } else if (converged) {
                LOG << "==> Coarsening converged";
                break;
            }
        }
    }

    if (!_ctx.debug.save_graph_hierarchy && _ctx.debug.save_coarsest_graph) {
        debug::save_graph(*graph, _ctx, static_cast<int>(coarsener.level()));
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Step 2: Initial Partitioning
    ////////////////////////////////////////////////////////////////////////////////
    if (mpi::get_comm_rank(_graph.communicator()) == 0) {
        cio::print_banner("Initial Partitioning");
    }

    auto initial_partitioner = TIMED_SCOPE("Allocation") {
        return factory::create_initial_partitioning_algorithm(_ctx);
    };

    START_TIMER("Initial Partitioning");
    auto                        shm_graph    = graph::allgather(*graph);
    auto                        shm_p_graph  = initial_partitioner->initial_partition(shm_graph);
    DistributedPartitionedGraph dist_p_graph = graph::reduce_scatter(*graph, std::move(shm_p_graph));
    STOP_TIMER();

    KASSERT(graph::debug::validate_partition(dist_p_graph), "", assert::heavy);

    const auto initial_cut       = metrics::edge_cut(dist_p_graph);
    const auto initial_imbalance = metrics::imbalance(dist_p_graph);

    LOG << "Initial partition: cut=" << initial_cut << " imbalance=" << initial_imbalance;

    if (_ctx.debug.save_imbalanced_partitions && initial_imbalance > _ctx.partition.epsilon) {
        debug::save_partitioned_graph(dist_p_graph, _ctx, static_cast<int>(coarsener.level()));
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Step 3: Refinement
    ////////////////////////////////////////////////////////////////////////////////
    {
        SCOPED_TIMER("Uncoarsening");
        auto ref_p_ctx = _ctx.partition;
        ref_p_ctx.setup(dist_p_graph.graph());

        if (mpi::get_comm_rank(_graph.communicator()) == 0) {
            cio::print_banner("Refinement");
        }

        auto refinement_algorithm = TIMED_SCOPE("Allocation") {
            return factory::create_refinement_algorithm(_ctx);
        };

        auto refine = [&](DistributedPartitionedGraph& p_graph) {
            {
                SCOPED_TIMER("Balancing");
                const bool feasible = metrics::is_feasible(p_graph, _ctx.partition);
                if (!feasible) {
                    const double imbalance = metrics::imbalance(p_graph);
                    LOG << "-> Balancing infeasible partition (imbalance=" << imbalance << ") ...";

                    DistributedBalancer balancer(_ctx);
                    balancer.initialize(p_graph);
                    balancer.balance(p_graph, _ctx.partition);
                }
            }
            {
                SCOPED_TIMER("Refinement");
                LOG << "-> Refining partition ...";
                refinement_algorithm->initialize(p_graph.graph(), _ctx.partition);
                refinement_algorithm->refine(p_graph);
                KASSERT(graph::debug::validate_partition(p_graph), "", assert::heavy);
            }
            {
                SCOPED_TIMER("Balancing");
                const bool feasible = metrics::is_feasible(p_graph, _ctx.partition);
                if (!feasible) {
                    const double imbalance = metrics::imbalance(p_graph);
                    LOG << "-> Balancing infeasible partition (imbalance=" << imbalance << ") ...";

                    DistributedBalancer balancer(_ctx);
                    balancer.initialize(p_graph);
                    balancer.balance(p_graph, _ctx.partition);
                }
            }
        };

        // special case: graph too small for multilevel, still run refinement
        if (_ctx.refinement.refine_coarsest_level) {
            SCOPED_TIMER("Uncoarsening", std::string("Level ") + std::to_string(coarsener.level()));
            ref_p_ctx.setup(dist_p_graph.graph());
            refine(dist_p_graph);

            // Output refinement statistics
            const auto current_cut       = metrics::edge_cut(dist_p_graph);
            const auto current_imbalance = metrics::imbalance(dist_p_graph);

            LOG << "=> level=" << coarsener.level() << " cut=" << current_cut << " imbalance=" << current_imbalance;

            if (_ctx.debug.save_imbalanced_partitions && current_imbalance > _ctx.partition.epsilon) {
                debug::save_partitioned_graph(dist_p_graph, _ctx, static_cast<int>(coarsener.level()));
            }
        }

        // Uncoarsen and refine
        while (coarsener.level() > 0) {
            SCOPED_TIMER("Uncoarsening", std::string("Level ") + std::to_string(coarsener.level()));

            dist_p_graph = TIMED_SCOPE("Uncontraction") {
                return coarsener.uncoarsen_once(std::move(dist_p_graph));
            };
            ref_p_ctx.setup(dist_p_graph.graph());

            refine(dist_p_graph);

            // Output refinement statistics
            const auto current_cut       = metrics::edge_cut(dist_p_graph);
            const auto current_imbalance = metrics::imbalance(dist_p_graph);

            LOG << "=> level=" << coarsener.level() << " cut=" << current_cut << " imbalance=" << current_imbalance;

            if (_ctx.debug.save_imbalanced_partitions && current_imbalance > _ctx.partition.epsilon) {
                debug::save_partitioned_graph(dist_p_graph, _ctx, static_cast<int>(coarsener.level()));
            }
        }
    }

    return dist_p_graph;
}
} // namespace kaminpar::dist