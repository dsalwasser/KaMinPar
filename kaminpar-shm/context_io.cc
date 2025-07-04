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

#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/console_io.h"
#include "kaminpar-common/random.h"
#include "kaminpar-common/strutils.h"

namespace kaminpar::shm {

using namespace std::string_literals;

std::unordered_map<std::string, NodeOrdering> get_node_orderings() {
  return {
      {"natural", NodeOrdering::NATURAL},
      {"deg-buckets", NodeOrdering::DEGREE_BUCKETS},
      {"degree-buckets", NodeOrdering::DEGREE_BUCKETS},
      {"external-deg-buckets", NodeOrdering::EXTERNAL_DEGREE_BUCKETS},
      {"external-degree-buckets", NodeOrdering::EXTERNAL_DEGREE_BUCKETS},
      {"implicit-deg-buckets", NodeOrdering::IMPLICIT_DEGREE_BUCKETS},
      {"implicit-degree-buckets", NodeOrdering::IMPLICIT_DEGREE_BUCKETS},
  };
}

std::ostream &operator<<(std::ostream &out, const NodeOrdering ordering) {
  switch (ordering) {
  case NodeOrdering::NATURAL:
    return out << "natural";
  case NodeOrdering::DEGREE_BUCKETS:
    return out << "deg-buckets";
  case NodeOrdering::EXTERNAL_DEGREE_BUCKETS:
    return out << "external-deg-buckets";
  case NodeOrdering::IMPLICIT_DEGREE_BUCKETS:
    return out << "implicit-deg-buckets";
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

std::unordered_map<std::string, CoarseningAlgorithm> get_coarsening_algorithms() {
  return {
      {"noop", CoarseningAlgorithm::NOOP},
      {"clustering", CoarseningAlgorithm::BASIC_CLUSTERING},
      {"basic-clustering", CoarseningAlgorithm::BASIC_CLUSTERING},
      {"overlay-clustering", CoarseningAlgorithm::OVERLAY_CLUSTERING},
      {"sparsification-clustering", CoarseningAlgorithm::SPARSIFICATION_CLUSTERING},
  };
}

std::ostream &operator<<(std::ostream &out, const CoarseningAlgorithm algorithm) {
  switch (algorithm) {
  case CoarseningAlgorithm::NOOP:
    return out << "noop";
  case CoarseningAlgorithm::BASIC_CLUSTERING:
    return out << "basic-clustering";
  case CoarseningAlgorithm::OVERLAY_CLUSTERING:
    return out << "overlay-clustering";
  case CoarseningAlgorithm::SPARSIFICATION_CLUSTERING:
    return out << "sparsification-clustering";
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

std::unordered_map<std::string, LabelPropagationImplementation> get_lp_implementations() {
  return {
      {"single-phase", LabelPropagationImplementation::SINGLE_PHASE},
      {"two-phase", LabelPropagationImplementation::TWO_PHASE},
      {"growing-hash-tables", LabelPropagationImplementation::GROWING_HASH_TABLES},
  };
}

std::ostream &operator<<(std::ostream &out, const LabelPropagationImplementation impl) {
  switch (impl) {
  case LabelPropagationImplementation::SINGLE_PHASE:
    return out << "single-phase";
  case LabelPropagationImplementation::TWO_PHASE:
    return out << "two-phase";
  case LabelPropagationImplementation::GROWING_HASH_TABLES:
    return out << "growing-hash-tables";
  }
  return out << "<invalid>";
}

std::unordered_map<std::string, RefinementAlgorithm> get_kway_refinement_algorithms() {
  return {
      {"noop", RefinementAlgorithm::NOOP},
      {"lp", RefinementAlgorithm::LABEL_PROPAGATION},
      {"fm", RefinementAlgorithm::KWAY_FM},
      {"jet", RefinementAlgorithm::JET},
      {"greedy-balancer", RefinementAlgorithm::OVERLOAD_BALANCER},
      {"overload-balancer", RefinementAlgorithm::OVERLOAD_BALANCER},
      {"underload-balancer", RefinementAlgorithm::UNDERLOAD_BALANCER},
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
  case RefinementAlgorithm::OVERLOAD_BALANCER:
    return out << "overload-balancer";
  case RefinementAlgorithm::UNDERLOAD_BALANCER:
    return out << "underload-balancer";
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
      {"vcycle", PartitioningMode::VCYCLE},
      {"rb", PartitioningMode::RB},
      {"kway", PartitioningMode::KWAY},
  };
}

std::ostream &operator<<(std::ostream &out, const PartitioningMode mode) {
  switch (mode) {
  case PartitioningMode::DEEP:
    return out << "deep";
  case PartitioningMode::VCYCLE:
    return out << "vcycle";
  case PartitioningMode::RB:
    return out << "rb";
  case PartitioningMode::KWAY:
    return out << "kway";
  }

  return out << "<invalid>";
}

std::unordered_map<std::string, DeepInitialPartitioningMode> get_deep_initial_partitioning_modes() {
  return {
      {"sequential", DeepInitialPartitioningMode::SEQUENTIAL},
      {"async-parallel", DeepInitialPartitioningMode::ASYNCHRONOUS_PARALLEL},
      {"sync-parallel", DeepInitialPartitioningMode::SYNCHRONOUS_PARALLEL},
      {"communities", DeepInitialPartitioningMode::COMMUNITIES},
  };
}

std::ostream &operator<<(std::ostream &out, const DeepInitialPartitioningMode mode) {
  switch (mode) {
  case DeepInitialPartitioningMode::SEQUENTIAL:
    return out << "sequential";
  case DeepInitialPartitioningMode::ASYNCHRONOUS_PARALLEL:
    return out << "async-parallel";
  case DeepInitialPartitioningMode::SYNCHRONOUS_PARALLEL:
    return out << "sync-parallel";
  case DeepInitialPartitioningMode::COMMUNITIES:
    return out << "communities";
  }

  return out << "<invalid>";
}

std::unordered_map<std::string, KwayInitialPartitioningMode> get_kway_initial_partitioning_modes() {
  return {
      {"sequential", KwayInitialPartitioningMode::SEQUENTIAL},
      {"parallel", KwayInitialPartitioningMode::PARALLEL},
      {"legacy", KwayInitialPartitioningMode::LEGACY},
  };
}

std::ostream &operator<<(std::ostream &out, const KwayInitialPartitioningMode mode) {
  switch (mode) {
  case KwayInitialPartitioningMode::SEQUENTIAL:
    return out << "sequential";
  case KwayInitialPartitioningMode::PARALLEL:
    return out << "parallel";
  case KwayInitialPartitioningMode::LEGACY:
    return out << "legacy";
  }

  return out << "<invalid>";
}

std::unordered_map<std::string, GainCacheStrategy> get_gain_cache_strategies() {
  return {
      {"compact-hashing", GainCacheStrategy::COMPACT_HASHING},
      {"compact-hashing-largek", GainCacheStrategy::COMPACT_HASHING_LARGE_K},
      {"sparse", GainCacheStrategy::SPARSE},
      {"sparse-largek", GainCacheStrategy::SPARSE_LARGE_K},
      {"hashing", GainCacheStrategy::HASHING},
      {"hashing-largek", GainCacheStrategy::HASHING_LARGE_K},
      {"dense", GainCacheStrategy::DENSE},
      {"dense-largek", GainCacheStrategy::DENSE_LARGE_K},
      {"on-the-fly", GainCacheStrategy::ON_THE_FLY},
  };
}

std::ostream &operator<<(std::ostream &out, const GainCacheStrategy strategy) {
  switch (strategy) {
  case GainCacheStrategy::COMPACT_HASHING:
    return out << "compact-hashing";
  case GainCacheStrategy::COMPACT_HASHING_LARGE_K:
    return out << "compact-hashing-largek";
  case GainCacheStrategy::SPARSE:
    return out << "sparse";
  case GainCacheStrategy::SPARSE_LARGE_K:
    return out << "sparse-largek";
  case GainCacheStrategy::HASHING:
    return out << "hashing";
  case GainCacheStrategy::HASHING_LARGE_K:
    return out << "hashing-largek";
  case GainCacheStrategy::DENSE:
    return out << "dense";
  case GainCacheStrategy::DENSE_LARGE_K:
    return out << "dense-largek";
  case GainCacheStrategy::ON_THE_FLY:
    return out << "on-the-fly";
  }

  return out << "<invalid>";
}

std::unordered_map<std::string, TieBreakingStrategy> get_tie_breaking_strategies() {
  return {
      {"geometric", TieBreakingStrategy::GEOMETRIC},
      {"uniform", TieBreakingStrategy::UNIFORM},
  };
}

std::ostream &operator<<(std::ostream &out, const TieBreakingStrategy strategy) {
  switch (strategy) {
  case TieBreakingStrategy::GEOMETRIC:
    return out << "geometric";
  case TieBreakingStrategy::UNIFORM:
    return out << "uniform";
  }

  return out << "<invalid>";
}

std::ostream &operator<<(std::ostream &out, const TwoHopStrategy strategy) {
  switch (strategy) {
  case TwoHopStrategy::DISABLE:
    return out << "disable";
  case TwoHopStrategy::MATCH:
    return out << "match";
  case TwoHopStrategy::MATCH_THREADWISE:
    return out << "match-threadwise";
  case TwoHopStrategy::CLUSTER:
    return out << "cluster";
  case TwoHopStrategy::CLUSTER_THREADWISE:
    return out << "cluster-threadwise";
  }

  return out << "<invalid>";
}

std::unordered_map<std::string, TwoHopStrategy> get_two_hop_strategies() {
  return {
      {"disable", TwoHopStrategy::DISABLE},
      {"match", TwoHopStrategy::MATCH},
      {"match-threadwise", TwoHopStrategy::MATCH_THREADWISE},
      {"cluster", TwoHopStrategy::CLUSTER},
      {"cluster-threadwise", TwoHopStrategy::CLUSTER_THREADWISE},
  };
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
    out << "Compression Scheme:           Gap Encoding + ";
    if (c_ctx.streamvbyte_encoding) {
      out << "StreamVByte Encoding\n";
    } else {
      out << "VarInt Encoding\n";
    }

    out << "  High Degree Encoding:       " << (c_ctx.high_degree_encoding ? "yes" : "no") << "\n";
    if (c_ctx.high_degree_encoding) {
      out << "    Threshold:                " << c_ctx.high_degree_threshold << "\n";
      out << "    Part Length:              " << c_ctx.high_degree_part_length << "\n";
    }
    out << "  Interval Encoding:          " << (c_ctx.interval_encoding ? "yes" : "no") << "\n";
    if (c_ctx.interval_encoding) {
      out << "    Length Threshold:         " << c_ctx.interval_length_treshold << "\n";
    }

    out << "Compresion Ratio:             " << c_ctx.compression_ratio
        << " [size reduction: " << (c_ctx.size_reduction / (float)(1024 * 1024)) << " mb]" << "\n";
    out << "  High Degree Node Count:     " << c_ctx.num_high_degree_nodes << "\n";
    out << "  High Degree Part Count:     " << c_ctx.num_high_degree_parts << "\n";
    out << "  Interval Node Count:        " << c_ctx.num_interval_nodes << "\n";
    out << "  Interval Count:             " << c_ctx.num_intervals << "\n";
  }
}

std::ostream &operator<<(std::ostream &out, const ContractionAlgorithm mode) {
  switch (mode) {
  case ContractionAlgorithm::BUFFERED:
    return out << "buffered";
  case ContractionAlgorithm::UNBUFFERED:
    return out << "unbuffered";
  case ContractionAlgorithm::UNBUFFERED_NAIVE:
    return out << "unbuffered-naive";
  }

  return out << "<invalid>";
}

std::unordered_map<std::string, ContractionAlgorithm> get_contraction_algorithms() {
  return {
      {"buffered", ContractionAlgorithm::BUFFERED},
      {"unbuffered", ContractionAlgorithm::UNBUFFERED},
      {"unbuffered-naive", ContractionAlgorithm::UNBUFFERED_NAIVE},
  };
}

std::ostream &operator<<(std::ostream &out, const ContractionImplementation mode) {
  switch (mode) {
  case ContractionImplementation::SINGLE_PHASE:
    return out << "single-phase";
  case ContractionImplementation::TWO_PHASE:
    return out << "two-phase";
  case ContractionImplementation::GROWING_HASH_TABLES:
    return out << "growing-hash-tables";
  }

  return out << "<invalid>";
}

std::unordered_map<std::string, ContractionImplementation> get_contraction_implementations() {
  return {
      {"single-phase", ContractionImplementation::SINGLE_PHASE},
      {"two-phase", ContractionImplementation::TWO_PHASE},
      {"growing-hash-tables", ContractionImplementation::GROWING_HASH_TABLES},
  };
}

void print(const CoarseningContext &c_ctx, std::ostream &out) {
  out << "Contraction limit:            " << c_ctx.contraction_limit << "\n";
  out << "Coarsening algorithm:         " << c_ctx.algorithm << "\n";

  if (c_ctx.algorithm == CoarseningAlgorithm::BASIC_CLUSTERING ||
      c_ctx.algorithm == CoarseningAlgorithm::OVERLAY_CLUSTERING ||
      c_ctx.algorithm == CoarseningAlgorithm::SPARSIFICATION_CLUSTERING) {
    out << "  Cluster weight limit:       " << c_ctx.clustering.cluster_weight_limit << " x "
        << c_ctx.clustering.cluster_weight_multiplier << "\n";
    out << "  Shrink factor:              " << c_ctx.clustering.shrink_factor << "\n";
    out << "  Max mem-free level:         " << c_ctx.clustering.max_mem_free_coarsening_level
        << "\n";
    out << "  Clustering algorithm:       " << c_ctx.clustering.algorithm << "\n";
    if (c_ctx.clustering.algorithm == ClusteringAlgorithm::LABEL_PROPAGATION) {
      print(c_ctx.clustering.lp, out);
    }
    out << "  Forced hierarchy levels:    " << (c_ctx.clustering.forced_kc_level ? "+kC " : "")
        << (c_ctx.clustering.forced_pc_level ? "+pC " : "")
        << ((!c_ctx.clustering.forced_kc_level && !c_ctx.clustering.forced_pc_level) ? "<none> "
                                                                                     : "")
        << "(leeway: U=" << c_ctx.clustering.forced_level_upper_factor
        << ", L=" << c_ctx.clustering.forced_level_lower_factor << ")\n";
  }
  if (c_ctx.algorithm == CoarseningAlgorithm::OVERLAY_CLUSTERING) {
    out << "  Overlays:" << "\n";
    out << "    Number of overlays:       " << (2 << c_ctx.overlay_clustering.num_levels) << "\n";
  }

  out << "Contraction algorithm:        " << c_ctx.contraction.algorithm << '\n';
  if (c_ctx.contraction.algorithm == ContractionAlgorithm::BUFFERED) {
    out << "  Edge buffer fill fraction:  " << c_ctx.contraction.edge_buffer_fill_fraction << "\n";
  } else if (c_ctx.contraction.algorithm == ContractionAlgorithm::UNBUFFERED) {
    out << "  Implementation:             " << c_ctx.contraction.unbuffered_implementation << "\n";
  }
}

void print(const LabelPropagationCoarseningContext &lp_ctx, std::ostream &out) {
  out << "    Number of iterations:     " << lp_ctx.num_iterations << "\n";
  out << "    High degree threshold:    " << lp_ctx.large_degree_threshold << "\n";
  out << "    Max degree:               " << lp_ctx.max_num_neighbors << "\n";
  out << "    Tie breaking strategy:    " << lp_ctx.tie_breaking_strategy << "\n";
  out << "    Implementation:           " << lp_ctx.impl << "\n";
  if (lp_ctx.impl == LabelPropagationImplementation::TWO_PHASE) {
    out << "      Relabel:                " << (lp_ctx.relabel_before_second_phase ? "yes" : "no")
        << '\n';
  }
  out << "    2-hop clustering:         " << lp_ctx.two_hop_strategy << ", if |Vcoarse| > "
      << std::setw(2) << std::fixed << lp_ctx.two_hop_threshold << " * |V|\n";
  out << "    Isolated nodes:           " << lp_ctx.isolated_nodes_strategy << "\n";
}

void print(const InitialPartitioningContext &i_ctx, std::ostream &out) {
  out << "Adaptive algorithm selection: "
      << (i_ctx.pool.use_adaptive_bipartitioner_selection ? "yes" : "no") << "\n";
}

void print(const RefinementContext &r_ctx, std::ostream &out) {
  out << "Refinement algorithms:        [" << str::implode(r_ctx.algorithms, " -> ") << "]\n";
  if (r_ctx.includes_algorithm(RefinementAlgorithm::LABEL_PROPAGATION)) {
    out << "Label propagation:\n";
    out << "  Number of iterations:       " << r_ctx.lp.num_iterations << "\n";
    out << "  Tie breaking strategy:      " << r_ctx.lp.tie_breaking_strategy << "\n";
    out << "  Implementation:             " << r_ctx.lp.impl << "\n";
  }
  if (r_ctx.includes_algorithm(RefinementAlgorithm::KWAY_FM)) {
    out << "k-way FM:\n";
    out << "  Number of iterations:       " << r_ctx.kway_fm.num_iterations
        << " [or improvement drops below < " << 100.0 * (1.0 - r_ctx.kway_fm.abortion_threshold)
        << "%]\n";
    out << "  Number of seed nodes:       " << r_ctx.kway_fm.num_seed_nodes << "\n";
    out << "  Locking strategies:         seed nodes: "
        << (r_ctx.kway_fm.unlock_seed_nodes ? "unlock" : "lock") << ", locally moved nodes: "
        << (r_ctx.kway_fm.unlock_locally_moved_nodes ? "unlock" : "lock") << "\n";
    out << "  Gain cache:                 " << r_ctx.kway_fm.gain_cache_strategy << "\n";
  }
  if (r_ctx.includes_algorithm(RefinementAlgorithm::JET)) {
    out << "Jet refinement:               " << RefinementAlgorithm::JET << "\n";
    out << "  Number of rounds:           coarse " << r_ctx.jet.num_rounds_on_coarse_level
        << ", fine " << r_ctx.jet.num_rounds_on_fine_level << "\n";
    out << "  Number of iterations:       max " << r_ctx.jet.num_iterations << ", or "
        << r_ctx.jet.num_fruitless_iterations << " fruitless (improvement < "
        << 100.0 * (1 - r_ctx.jet.fruitless_threshold) << "%)\n";
    out << "  Gain temperature:           coarse [" << r_ctx.jet.initial_gain_temp_on_coarse_level
        << ", " << r_ctx.jet.final_gain_temp_on_coarse_level << "], " << "fine ["
        << r_ctx.jet.initial_gain_temp_on_fine_level << ", "
        << r_ctx.jet.final_gain_temp_on_fine_level << "]\n";
    out << "  Balancing algorithm:        " << r_ctx.jet.balancing_algorithm << "\n";
  }
}

void print(const PartitionContext &p_ctx, std::ostream &out) {
  // @todo rework block weights output
  const auto max_block_weight = static_cast<std::int64_t>(p_ctx.max_block_weight(0));
  const auto size = std::max<std::int64_t>(
      {static_cast<std::int64_t>(p_ctx.n), static_cast<std::int64_t>(p_ctx.m), max_block_weight}
  );
  const std::size_t width = size > 0 ? std::ceil(std::log10(size)) : 1;

  out << "  Number of nodes:            " << std::setw(width) << p_ctx.n;
  if (static_cast<NodeWeight>(p_ctx.n) == p_ctx.total_node_weight) {
    out << " (unweighted)\n";
  } else {
    out << " (total weight: " << p_ctx.total_node_weight << ")\n";
  }
  out << "  Number of edges:            " << std::setw(width) << p_ctx.m;
  if (static_cast<EdgeWeight>(p_ctx.m) == p_ctx.total_edge_weight) {
    out << " (unweighted)\n";
  } else {
    out << " (total weight: " << p_ctx.total_edge_weight << ")\n";
  }
  out << "Number of blocks:             " << p_ctx.k << "\n";
  out << "Maximum block weight:         " << p_ctx.max_block_weight(0) << " ("
      << p_ctx.perfectly_balanced_block_weight(0) << " + " << 100 * p_ctx.epsilon() << "% / "
      << 100 * p_ctx.inferred_epsilon() << "%)\n";
}

void print(const PartitioningContext &p_ctx, std::ostream &out) {
  out << "Partitioning mode:            " << p_ctx.mode << "\n";
  if (p_ctx.mode == PartitioningMode::DEEP) {
    out << "  Deep initial part. mode:    " << p_ctx.deep_initial_partitioning_mode << "\n";
    out << "  Deep initial part. load:    " << p_ctx.deep_initial_partitioning_load << "\n";
  } else if (p_ctx.mode == PartitioningMode::KWAY) {
    out << "  Initial partitioning mode:  " << p_ctx.kway_initial_partitioning_mode << "\n";
  } else if (p_ctx.mode == PartitioningMode::RB) {
    out << "  Use flat k-way refinement:  "
        << (p_ctx.rb_enable_kway_toplevel_refinement ? "yes" : "no") << "\n";
    out << "  Switch to seq. part.:       "
        << (p_ctx.rb_switch_to_seq_factor == 0
                ? "never"
                : "when k' > p * " + std::to_string(p_ctx.rb_switch_to_seq_factor))
        << "\n";
  }
  out << "Subgraph memory:              " << (p_ctx.use_lazy_subgraph_memory ? "Lazy" : "Default")
      << "\n";
}

void print(const Context &ctx, std::ostream &out) {
  out << "Execution mode:               " << ctx.parallel.num_threads << "\n";
  out << "Seed:                         " << Random::get_seed() << "\n";
  out << "Graph:                        " << ctx.debug.graph_name
      << " [node ordering: " << ctx.node_ordering << "]" << " [edge ordering: " << ctx.edge_ordering
      << "]\n";
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
