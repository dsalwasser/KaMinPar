/*******************************************************************************
 * IO functions for the context structs.
 *
 * @file:   context_io.cc
 * @author: Daniel Seemaier
 * @date:   13.03.2023
 ******************************************************************************/
#include "kaminpar-shm/context_io.h"

#include <algorithm>
#include <cmath>
#include <iomanip>

#include "kaminpar-common/asserting_cast.h"
#include "kaminpar-common/console_io.h"
#include "kaminpar-common/random.h"
#include "kaminpar-common/strutils.h"
#include "kaminpar-common/varint_codec.h"

namespace kaminpar::shm {
using namespace std::string_literals;

std::unordered_map<std::string, NodeOrdering> get_node_orderings() {
  return {
      {"natural", NodeOrdering::NATURAL},
      {"deg-buckets", NodeOrdering::DEGREE_BUCKETS},
      {"degree-buckets", NodeOrdering::DEGREE_BUCKETS},
  };
}

std::ostream &operator<<(std::ostream &out, const NodeOrdering ordering) {
  switch (ordering) {
  case NodeOrdering::NATURAL:
    return out << "natural";
  case NodeOrdering::DEGREE_BUCKETS:
    return out << "deg-buckets";
  }

  return out << "<invalid>";
}

std::unordered_map<std::string, EdgeOrdering> get_edge_orderings() {
  return {
      {"natural", EdgeOrdering::NATURAL},
      {"compression", EdgeOrdering::COMPRESSION},
  };
}

std::ostream &operator<<(std::ostream &out, const EdgeOrdering ordering) {
  switch (ordering) {
  case EdgeOrdering::NATURAL:
    return out << "natural";
  case EdgeOrdering::COMPRESSION:
    return out << "compression";
  }

  return out << "<invalid>";
}

std::unordered_map<std::string, ClusteringAlgorithm> get_clustering_algorithms() {
  return {
      {"noop", ClusteringAlgorithm::NOOP},
      {"lp", ClusteringAlgorithm::LABEL_PROPAGATION},
  };
}

std::ostream &operator<<(std::ostream &out, const ClusteringAlgorithm algorithm) {
  switch (algorithm) {
  case ClusteringAlgorithm::NOOP:
    return out << "noop";
  case ClusteringAlgorithm::LABEL_PROPAGATION:
    return out << "lp";
  }
  return out << "<invalid>";
}

std::unordered_map<std::string, ClusterWeightLimit> get_cluster_weight_limits() {
  return {
      {"epsilon-block-weight", ClusterWeightLimit::EPSILON_BLOCK_WEIGHT},
      {"static-block-weight", ClusterWeightLimit::BLOCK_WEIGHT},
      {"one", ClusterWeightLimit::ONE},
      {"zero", ClusterWeightLimit::ZERO},
  };
}

std::ostream &operator<<(std::ostream &out, const ClusterWeightLimit limit) {
  switch (limit) {
  case ClusterWeightLimit::EPSILON_BLOCK_WEIGHT:
    return out << "epsilon-block-weight";
  case ClusterWeightLimit::BLOCK_WEIGHT:
    return out << "static-block-weight";
  case ClusterWeightLimit::ONE:
    return out << "one";
  case ClusterWeightLimit::ZERO:
    return out << "zero";
  }
  return out << "<invalid>";
}

std::unordered_map<std::string, RefinementAlgorithm> get_kway_refinement_algorithms() {
  return {
      {"noop", RefinementAlgorithm::NOOP},
      {"lp", RefinementAlgorithm::LABEL_PROPAGATION},
      {"fm", RefinementAlgorithm::KWAY_FM},
      {"jet", RefinementAlgorithm::JET},
      {"greedy-balancer", RefinementAlgorithm::GREEDY_BALANCER},
      {"mtkahypar", RefinementAlgorithm::MTKAHYPAR},
  };
}

std::ostream &operator<<(std::ostream &out, const RefinementAlgorithm algorithm) {
  switch (algorithm) {
  case RefinementAlgorithm::NOOP:
    return out << "noop";
  case RefinementAlgorithm::KWAY_FM:
    return out << "fm";
  case RefinementAlgorithm::LABEL_PROPAGATION:
    return out << "lp";
  case RefinementAlgorithm::GREEDY_BALANCER:
    return out << "greedy-balancer";
  case RefinementAlgorithm::JET:
    return out << "jet";
  case RefinementAlgorithm::MTKAHYPAR:
    return out << "mtkahypar";
  }

  return out << "<invalid>";
}

std::unordered_map<std::string, FMStoppingRule> get_fm_stopping_rules() {
  return {
      {"simple", FMStoppingRule::SIMPLE},
      {"adaptive", FMStoppingRule::ADAPTIVE},
  };
}

std::ostream &operator<<(std::ostream &out, const FMStoppingRule rule) {
  switch (rule) {
  case FMStoppingRule::SIMPLE:
    return out << "simple";
  case FMStoppingRule::ADAPTIVE:
    return out << "adaptive";
  }

  return out << "<invalid>";
}

std::unordered_map<std::string, PartitioningMode> get_partitioning_modes() {
  return {
      {"deep", PartitioningMode::DEEP},
      {"rb", PartitioningMode::RB},
      {"kway", PartitioningMode::KWAY},
  };
}

std::ostream &operator<<(std::ostream &out, const PartitioningMode mode) {
  switch (mode) {
  case PartitioningMode::DEEP:
    return out << "deep";
  case PartitioningMode::RB:
    return out << "rb";
  case PartitioningMode::KWAY:
    return out << "kway";
  }

  return out << "<invalid>";
}

std::unordered_map<std::string, InitialPartitioningMode> get_initial_partitioning_modes() {
  return {
      {"sequential", InitialPartitioningMode::SEQUENTIAL},
      {"async-parallel", InitialPartitioningMode::ASYNCHRONOUS_PARALLEL},
      {"sync-parallel", InitialPartitioningMode::SYNCHRONOUS_PARALLEL},
  };
}

std::ostream &operator<<(std::ostream &out, const InitialPartitioningMode mode) {
  switch (mode) {
  case InitialPartitioningMode::SEQUENTIAL:
    return out << "sequential";
  case InitialPartitioningMode::ASYNCHRONOUS_PARALLEL:
    return out << "async-parallel";
  case InitialPartitioningMode::SYNCHRONOUS_PARALLEL:
    return out << "sync-parallel";
  }

  return out << "<invalid>";
}

std::unordered_map<std::string, GainCacheStrategy> get_gain_cache_strategies() {
  return {
      {"dense", GainCacheStrategy::DENSE},
      {"on-the-fly", GainCacheStrategy::ON_THE_FLY},
      {"hybrid", GainCacheStrategy::HYBRID},
  };
}

std::ostream &operator<<(std::ostream &out, const GainCacheStrategy strategy) {
  switch (strategy) {
  case GainCacheStrategy::DENSE:
    return out << "dense";
  case GainCacheStrategy::ON_THE_FLY:
    return out << "on-the-fly";
  case GainCacheStrategy::HYBRID:
    return out << "hybrid";
  }

  return out << "<invalid>";
}

std::ostream &operator<<(std::ostream &out, IsolatedNodesClusteringStrategy strategy) {
  switch (strategy) {
  case IsolatedNodesClusteringStrategy::KEEP:
    return out << "keep";
  case IsolatedNodesClusteringStrategy::MATCH:
    return out << "match";
  case IsolatedNodesClusteringStrategy::CLUSTER:
    return out << "cluster";
  case IsolatedNodesClusteringStrategy::MATCH_DURING_TWO_HOP:
    return out << "match-during-two-hop";
  case IsolatedNodesClusteringStrategy::CLUSTER_DURING_TWO_HOP:
    return out << "cluster-during-two-hop";
  }

  return out << "<invalid>";
}

std::unordered_map<std::string, IsolatedNodesClusteringStrategy>
get_isolated_nodes_clustering_strategies() {
  return {
      {"keep", IsolatedNodesClusteringStrategy::KEEP},
      {"match", IsolatedNodesClusteringStrategy::MATCH},
      {"cluster", IsolatedNodesClusteringStrategy::CLUSTER},
      {"match-during-two-hop", IsolatedNodesClusteringStrategy::MATCH_DURING_TWO_HOP},
      {"cluster-during-two-hop", IsolatedNodesClusteringStrategy::CLUSTER_DURING_TWO_HOP},
  };
}

void print(const GraphCompressionContext &c_ctx, std::ostream &out) {
  out << "Enabled:                      " << (c_ctx.enabled ? "yes" : "no") << "\n";
  if (c_ctx.enabled) {
    out << "Compression Scheme:           "
        << "Gap Encoding + ";
    if (c_ctx.run_length_encoding) {
      out << "VarInt Run-Length Encoding\n";
    } else if (c_ctx.stream_encoding) {
      out << "VarInt Stream Encoding\n";
    } else {
      out << "VarInt Encoding\n";
    }
    out << "  High Degree Encoding:       " << (c_ctx.high_degree_encoding ? "yes" : "no") << "\n";
    out << "  High Degree Threshold:      " << c_ctx.high_degree_threshold << "\n";
    out << "  High Degree Part Length:    " << c_ctx.high_degree_part_length << "\n";
    out << "  Interval Encoding:          " << (c_ctx.interval_encoding ? "yes" : "no") << "\n";
    out << "  Interval Length Threshold:  " << c_ctx.interval_length_treshold << "\n";
    out << "  Isolated Nodes Separation:  " << (c_ctx.isolated_nodes_separation ? "yes" : "no")
        << "\n";

    out << "Compresion Ratio:             " << c_ctx.compression_ratio
        << " [size reduction: " << (c_ctx.size_reduction / (float)(1024 * 1024)) << " mb]"
        << "\n";
    out << "  High Degree Count:          " << c_ctx.high_degree_count << "\n";
    out << "  Part Count:                 " << c_ctx.part_count << "\n";
    out << "  Interval Count:             " << c_ctx.interval_count << "\n";

    if (debug::kTrackVarintStats) {
      const auto &stats = debug::varint_stats_global();

      const float avg_varint_len =
          (stats.varint_count == 0) ? 0 : (stats.varint_bytes / (float)stats.varint_count);
      out << "Average Varint Length:        " << avg_varint_len << " [count: " << stats.varint_count
          << "]\n";

      const float avg_signed_varint_len =
          (stats.signed_varint_count == 0)
              ? 0
              : (stats.signed_varint_bytes / (float)stats.signed_varint_count);
      out << "Average Signed Varint Length: " << avg_signed_varint_len
          << " [count: " << stats.signed_varint_count << "]\n";

      const float avg_marked_varint_len =
          (stats.marked_varint_count == 0)
              ? 0
              : (stats.marked_varint_bytes / (float)stats.marked_varint_count);
      out << "Average Marked Varint Length: " << avg_marked_varint_len
          << " [count: " << stats.marked_varint_count << "]\n";
    }
  }
}

void print(const CoarseningContext &c_ctx, std::ostream &out) {
  out << "Contraction limit:            " << c_ctx.contraction_limit << "\n";
  out << "Cluster weight limit:         " << c_ctx.cluster_weight_limit << " x "
      << c_ctx.cluster_weight_multiplier << "\n";
  out << "Clustering algorithm:         " << c_ctx.algorithm << "\n";
  if (c_ctx.algorithm == ClusteringAlgorithm::LABEL_PROPAGATION) {
    print(c_ctx.lp, out);
  }
}

void print(const LabelPropagationCoarseningContext &lp_ctx, std::ostream &out) {
  out << "  Number of iterations:       " << lp_ctx.num_iterations << "\n";
  out << "  High degree threshold:      " << lp_ctx.large_degree_threshold << "\n";
  out << "  Max degree:                 " << lp_ctx.max_num_neighbors << "\n";
  out << "  2-hop clustering threshold: " << std::fixed << 100 * lp_ctx.two_hop_clustering_threshold
      << "%\n";
  out << "  Isolated nodes:             " << lp_ctx.isolated_nodes_strategy << "\n";
  out << "  Uses two phases: " << (lp_ctx.use_two_phases ? "yes" : "no") << "\n";
}

void print(const InitialPartitioningContext &i_ctx, std::ostream &out) {
  out << "Adaptive algorithm selection: "
      << (i_ctx.use_adaptive_bipartitioner_selection ? "yes" : "no") << "\n";
}

void print(const RefinementContext &r_ctx, std::ostream &out) {
  out << "Refinement algorithms:        [" << str::implode(r_ctx.algorithms, " -> ") << "]\n";
  if (r_ctx.includes_algorithm(RefinementAlgorithm::LABEL_PROPAGATION)) {
    out << "Label propagation:\n";
    out << "  Number of iterations:       " << r_ctx.lp.num_iterations << "\n";
  }
  if (r_ctx.includes_algorithm(RefinementAlgorithm::KWAY_FM)) {
    out << "k-way FM:\n";
    out << "  Number of iterations:       " << r_ctx.kway_fm.num_iterations
        << " [or improvement drops below < " << 100.0 * (1.0 - r_ctx.kway_fm.abortion_threshold)
        << "%]\n";
    out << "  Number of seed nodes:       " << r_ctx.kway_fm.num_seed_nodes << "\n";
    out << "  Gain cache:                 " << r_ctx.kway_fm.gain_cache_strategy << "\n";
    if (r_ctx.kway_fm.gain_cache_strategy == GainCacheStrategy::HYBRID) {
      out << "  High-degree threshold:\n";
      out << "    based on k:               " << r_ctx.kway_fm.k_based_high_degree_threshold
          << "\n";
      out << "    constant:                 " << r_ctx.kway_fm.constant_high_degree_threshold
          << "\n";
      out << "  Preallocate gain cache:     " << r_ctx.kway_fm.preallocate_gain_cache << "\n";
    }
  }
}

void print(const PartitionContext &p_ctx, std::ostream &out) {
  const std::int64_t max_block_weight = p_ctx.block_weights.max(0);
  const std::int64_t size = std::max<std::int64_t>({p_ctx.n, p_ctx.m, max_block_weight});
  const std::size_t width = std::ceil(std::log10(size));

  out << "  Number of nodes:            " << std::setw(width) << p_ctx.n;
  if (asserting_cast<NodeWeight>(p_ctx.n) == p_ctx.total_node_weight) {
    out << " (unweighted)\n";
  } else {
    out << " (total weight: " << p_ctx.total_node_weight << ")\n";
  }
  out << "  Number of edges:            " << std::setw(width) << p_ctx.m;
  if (asserting_cast<EdgeWeight>(p_ctx.m) == p_ctx.total_edge_weight) {
    out << " (unweighted)\n";
  } else {
    out << " (total weight: " << p_ctx.total_edge_weight << ")\n";
  }
  out << "Number of blocks:             " << p_ctx.k << "\n";
  out << "Maximum block weight:         " << p_ctx.block_weights.max(0) << " ("
      << p_ctx.block_weights.perfectly_balanced(0) << " + " << 100 * p_ctx.epsilon << "%)\n";
}

void print(const PartitioningContext &p_ctx, std::ostream &out) {
  out << "Partitioning mode:            " << p_ctx.mode << "\n";
  if (p_ctx.mode == PartitioningMode::DEEP) {
    out << "  Deep initial part. mode:    " << p_ctx.deep_initial_partitioning_mode << "\n";
    out << "  Deep initial part. load:    " << p_ctx.deep_initial_partitioning_load << "\n";
  }
}

void print(const Context &ctx, std::ostream &out) {
  out << "Execution mode:               " << ctx.parallel.num_threads << "\n";
  out << "Seed:                         " << Random::get_seed() << "\n";
  out << "Graph:                        " << ctx.debug.graph_name
      << " [node ordering: " << ctx.node_ordering << "]"
      << " [edge ordering: " << ctx.edge_ordering << "]\n";
  print(ctx.partition, out);
  cio::print_delimiter("Graph Compression", '-');
  print(ctx.compression, out);
  cio::print_delimiter("Partitioning Scheme", '-');
  print(ctx.partitioning, out);
  cio::print_delimiter("Coarsening", '-');
  print(ctx.coarsening, out);
  cio::print_delimiter("Initial Partitioning", '-');
  print(ctx.initial_partitioning, out);
  cio::print_delimiter("Refinement", '-');
  print(ctx.refinement, out);
}
} // namespace kaminpar::shm
