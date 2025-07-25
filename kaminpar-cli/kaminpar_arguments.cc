/*******************************************************************************
 * Command line arguments for the shared-memory partitioner.
 *
 * @file:   kaminpar_arguments.cc
 * @author: Daniel Seemaier
 * @date:   14.10.2022
 ******************************************************************************/
#include "kaminpar-cli/kaminpar_arguments.h"

#include "kaminpar-cli/CLI11.h"

#include "kaminpar-shm/context.h"
#include "kaminpar-shm/kaminpar.h"

namespace kaminpar::shm {

void create_all_options(CLI::App *app, Context &ctx) {
  create_graph_compression_options(app, ctx);
  create_partitioning_options(app, ctx);
  create_debug_options(app, ctx);
  create_coarsening_options(app, ctx);
  create_initial_partitioning_options(app, ctx);
  create_refinement_options(app, ctx);
}

CLI::Option_group *create_graph_compression_options(CLI::App *app, Context &ctx) {
  auto *compression = app->add_option_group("Graph Compression");

  compression->add_flag("-c,--compress", ctx.compression.enabled, "Enable graph compression.")
      ->capture_default_str();

  return compression;
}

CLI::Option_group *create_partitioning_options(CLI::App *app, Context &ctx) {
  auto *partitioning = app->add_option_group("Partitioning");

  // Partitioning options
  partitioning->add_option("-m,--p-mode", ctx.partitioning.mode)
      ->transform(CLI::CheckedTransformer(get_partitioning_modes()).description(""))
      ->description(
          R"(Chooses the partitioning scheme:
  - deep: Deep multilevel (also see --p-deep-* options).
  - rb:   Recursive multilevel bipartitioning (also see --p-rb-* options).
  - kway: k-way multilevel, initial partitioning via recursive bipartitioning (also see --p-kway-* options).)"
      )
      ->capture_default_str();

  partitioning
      ->add_option(
          "--p-deep-initial-partitioning-mode", ctx.partitioning.deep_initial_partitioning_mode
      )
      ->transform(CLI::CheckedTransformer(get_deep_initial_partitioning_modes()).description(""))
      ->description(
          R"(Chooses the initial partitioning mode:
  - sequential:     Do not diversify initial partitioning by replicating coarse graphs.
  - async-parallel: Each branch of the replication tree proceeds asynchronously.
  - sync-parallel:  All branches proceed synchronously, i.e., perform coarsening, initial partitioning and refinement in a lock-step.)"
      )
      ->capture_default_str();
  partitioning->add_option(
      "--p-deep-initial-partitioning-load",
      ctx.partitioning.deep_initial_partitioning_load,
      "Controls the fraction of cores that should be assumed during the coarse graph replication "
      "phase of deep MGP: a value of '1' means that there will be <number of PEs> replications of "
      "the coarsest graph."
  );
  partitioning
      ->add_flag(
          "--p-deep-refine-after-extending",
          ctx.partitioning.refine_after_extending_partition,
          "If set, additional k-way refinement will be performed between subdividing partition "
          "blocks and uncontracting the coarse graph. Might improve partition quality slightly but "
          "costs additional running time."
      )
      ->capture_default_str();

  partitioning
      ->add_flag(
          "--p-rb-kway-toplevel-refinement",
          ctx.partitioning.rb_enable_kway_toplevel_refinement,
          "If set, performs additional k-way refinement on the toplevel (when using the recursive "
          "bipartitioning scheme)."
      )
      ->capture_default_str();

  partitioning
      ->add_flag(
          "--p-kway-initial-partitioning-mode", ctx.partitioning.kway_initial_partitioning_mode
      )
      ->transform(CLI::CheckedTransformer(get_kway_initial_partitioning_modes()).description(""))
      ->description(R"(Chooses the initial partitioning mode when using k-way partitioning:
  - sequential
  - parallel
  - legacy)")
      ->capture_default_str();

  partitioning
      ->add_option(
          "--p-vcycles",
          ctx.partitioning.vcycles,
          "Configures v-cycles: this option expects a vector <k0 <= k1 <= ... <= k> of "
          "non-decresing integers smaller or equal to k. For each integer ki, one v-cycle will be "
          "performed with k set to ki. For instance, if the sequence is 2, 4, 16, 16, the graph "
          "will first be partitioned into 2 blocks, then 4, then 16, then 16 (standard v-cycle). "
          "See also --p-vcycle-* for further options."
      )
      ->capture_default_str();
  partitioning
      ->add_flag(
          "--p-vcycle-restrict-refinement",
          ctx.partitioning.restrict_vcycle_refinement,
          "If set, nodes may not be moved outside their current block in further v-cycles. For "
          "instance, when using this option in combination with `--p-vcycles=2 8`, the graph will "
          "first be partitioned into 2 blocks; afterwards (in the next v-cycle), each block will "
          "be further partitioned into 4 more blocks (for a total of 8 blocks). In this step, "
          "nodes may not be moved into the other block of the original bipartition."
      )
      ->capture_default_str();

  create_partitioning_rearrangement_options(app, ctx);

  return partitioning;
}

CLI::Option_group *create_partitioning_rearrangement_options(CLI::App *app, Context &ctx) {
  auto *rearrangement = app->add_option_group("Partitioning -> Rearrangement");

  rearrangement->add_option("--node-order", ctx.node_ordering)
      ->transform(CLI::CheckedTransformer(get_node_orderings()).description(""))
      ->description(R"(Criteria by which the nodes of the graph are sorted and rearranged:
  - natural:              Keep the node order of the graph (do not rearrange).
  - deg-buckets:          Sort the nodes by degree bucket and rearrange accordingly.
  - external-deg-buckets: Sort the nodes by degree bucket and rearrange accordingly during IO (slower IO, but reduces RAM usage).
  - implicit-deg-buckets: Nodes of the input graph are already stored in degree bucket order.)")
      ->capture_default_str();
  rearrangement->add_option("--edge-order", ctx.edge_ordering)
      ->transform(CLI::CheckedTransformer(get_edge_orderings()).description(""))
      ->description(
          R"(Criteria by which the edges of the graph are sorted and rearranged (for debugging purposes):
  - natural:     Keep the edge order of the graph (do not rearrange).
  - compression: Sort the edges of each neighbourhood with the ordering of the corresponding compressed graph.)"
      )
      ->capture_default_str();

  return rearrangement;
}

CLI::Option_group *create_coarsening_options(CLI::App *app, Context &ctx) {
  auto *coarsening = app->add_option_group("Coarsening");

  // Coarsening options:
  coarsening->add_option("--c-algorithm", ctx.coarsening.algorithm)
      ->transform(CLI::CheckedTransformer(get_coarsening_algorithms()).description(""))
      ->description(
          R"(Chooses the coarsening algorithm:
  - noop:                      Disable coarsening (for debugging purposes).
  - clustering:                Perform coarsening via cluster contraction.
  - overlay-clustering:        Perform coarsening via contracting an ensemble of clusterings (also see --c-overlay-* options).
  - sparsification-clustering: Perform coarsening via cluster contraction with additional edge sparsification (also see --c-sparsification-* options).
For all *-clustering algorithms, also see the --c-clustering-* options.)"
      )
      ->capture_default_str();

  coarsening
      ->add_option(
          "--c-contraction-limit",
          ctx.coarsening.contraction_limit,
          "Upper limit for the number of nodes per block in the coarsest graph."
      )
      ->capture_default_str();

  coarsening
      ->add_option(
          "--c-convergence-threshold",
          ctx.coarsening.convergence_threshold,
          "Coarsening converges once the size of the graph shrinks by less than this factor."
      )
      ->capture_default_str();

  // Clustering options:
  coarsening
      ->add_option(
          "--c-shrink-factor",
          ctx.coarsening.clustering.shrink_factor,
          "Upper limit on how fast the graph can shrink."
      )
      ->capture_default_str();

  coarsening->add_option("--c-clustering-algorithm", ctx.coarsening.clustering.algorithm)
      ->transform(CLI::CheckedTransformer(get_clustering_algorithms()).description(""))
      ->description(R"(Chooses the clustering algorithm:
  - noop: Disable coarsening (for debugging purposes).
  - lp:   Size-constrained label propagation (also see --c-lp-* options).)")
      ->capture_default_str();

  coarsening->add_option("--c-cluster-weight-limit", ctx.coarsening.clustering.cluster_weight_limit)
      ->transform(CLI::CheckedTransformer(get_cluster_weight_limits()).description(""))
      ->description(
          R"(Chooses the formula used to compute the maximum cluster weight:
  - epsilon-block-weight: Cmax = eps * c(V) * min{n' / C, k}, where n' is the number of nodes in the current (coarse) graph.
  - static-block-weight:  Cmax = c(V) / k.
  - one:                  Cmax = 1.
  - zero:                 Cmax = 0 (disable coarsening).
Also see --c-cluster-weight-multiplier.)"
      )
      ->capture_default_str();
  coarsening
      ->add_option(
          "--c-cluster-weight-multiplier",
          ctx.coarsening.clustering.cluster_weight_multiplier,
          "Multiplicator of the maximum cluster weight base value."
      )
      ->capture_default_str();

  coarsening
      ->add_option(
          "--c-max-memory-free-coarsening-level",
          ctx.coarsening.clustering.max_mem_free_coarsening_level,
          "Maximum coarsening level for which the corresponding memory should be released "
          "afterwards."
      )
      ->capture_default_str();

  coarsening
      ->add_flag(
          "--c-forced-kc-level",
          ctx.coarsening.clustering.forced_kc_level,
          "Forces a coarsening level with roughly kC nodes."
      )
      ->capture_default_str();
  coarsening
      ->add_flag(
          "--c-forced-pc-level",
          ctx.coarsening.clustering.forced_pc_level,
          "Forces a coarsening level with roughly pC nodes."
      )
      ->capture_default_str();

  coarsening
      ->add_option(
          "--c-overlay-levels",
          ctx.coarsening.overlay_clustering.num_levels,
          "Use 2^<arg> clusterings in the ensemble."
      )
      ->capture_default_str();
  coarsening
      ->add_option(
          "--c-overlay-max-level",
          ctx.coarsening.overlay_clustering.max_level,
          "Only use cluster ensembles on the first <arg> coarsening levels."
      )
      ->capture_default_str();

  coarsening
      ->add_option(
          "--c-sparsification-density-factor",
          ctx.coarsening.sparsification_clustering.density_target_factor
      )
      ->description(
          "By which factor the density should at least be reduced from one level to the next: new "
          "density <= factor * old density. The default is infinity."
      )
      ->capture_default_str();
  coarsening
      ->add_option(
          "--c-sparsification-edge-factor",
          ctx.coarsening.sparsification_clustering.edge_target_factor
      )
      ->description(
          "By which factor the number of edges should at least be reduced from one level to the "
          "next: new number of edges <= factor * old number of edges The default is infinity."
      )
      ->capture_default_str();
  coarsening
      ->add_option(
          "--c-sparsification-laziness-factor",
          ctx.coarsening.sparsification_clustering.laziness_factor
      )
      ->description(
          "Only sparsify if the number of edges times the laziness factor are less than the "
          "target. Thus greater laziness leads to less sparsification. The default is 1."
      )
      ->capture_default_str();

  create_lp_coarsening_options(app, ctx);
  create_contraction_coarsening_options(app, ctx);

  return coarsening;
}

CLI::Option_group *create_lp_coarsening_options(CLI::App *app, Context &ctx) {
  auto *lp = app->add_option_group("Coarsening -> Label Propagation");

  lp->add_option(
        "--c-lp-num-iterations",
        ctx.coarsening.clustering.lp.num_iterations,
        "Number of label propagation iterations."
  )
      ->capture_default_str();
  lp->add_option(
        "--c-lp-active-large-degree-threshold",
        ctx.coarsening.clustering.lp.large_degree_threshold,
        "Threshold for ignoring nodes with large degree."
  )
      ->capture_default_str();
  lp->add_option(
        "--c-lp-max-num-neighbors",
        ctx.coarsening.clustering.lp.max_num_neighbors,
        "Limit the neighborhood to this many nodes."
  )
      ->capture_default_str();

  lp->add_option("--c-lp-tie-breaking-strategy", ctx.coarsening.clustering.lp.tie_breaking_strategy)
      ->transform(CLI::CheckedTransformer(get_tie_breaking_strategies()).description(""))
      ->description(
          R"(Chooses the tie breaking strategy:
  - geometric: Prefer nodes with same rating located at the end of a neighborhood.
  - uniform:   Select nodes with same rating uniformly at random.
  )"
      )
      ->capture_default_str();
  lp->add_option("--c-lp-impl", ctx.coarsening.clustering.lp.impl)
      ->transform(CLI::CheckedTransformer(get_lp_implementations()).description(""))
      ->description(
          R"(Chooses the label propagation implementation:
  - single-phase:        Uses single-phase label propagation.
  - two-phase:           Uses two-phase label propagation.
  - growing-hash-tables: Uses single-phase label propagation with growing hash tables.
  )"
      )
      ->capture_default_str();

  lp->add_option(
        "--c-lp-second-phase-relabel",
        ctx.coarsening.clustering.lp.relabel_before_second_phase,
        "Relabel the clusters before running the second phase."
  )
      ->capture_default_str();

  lp->add_option("--c-lp-two-hop-strategy", ctx.coarsening.clustering.lp.two_hop_strategy)
      ->transform(CLI::CheckedTransformer(get_two_hop_strategies()).description(""))
      ->description(R"(Chooses the strategy for handling singleton clusters during coarsening:
  - disable: Do not merge two-hop singleton clusters.
  - match:   Join two-hop singleton clusters pairwise.
  - cluster: Cluster two-hop singleton clusters into a single cluster (respecting the maximum cluster weight limit).
  )")
      ->capture_default_str();
  lp->add_option(
        "--c-lp-two-hop-threshold",
        ctx.coarsening.clustering.lp.two_hop_threshold,
        "Enable two-hop clustering if plain label propagation shrunk "
        "the graph by less than this factor."
  )
      ->capture_default_str();

  lp->add_option(
        "--c-lp-isolated-nodes-strategy", ctx.coarsening.clustering.lp.isolated_nodes_strategy
  )
      ->transform(
          CLI::CheckedTransformer(get_isolated_nodes_clustering_strategies()).description("")
      )
      ->description(R"(Chooses the strategy for handling isolated nodes during graph clustering:
  - keep:                   Keep isolated nodes in the graph.
  - match-always:           Pack pairs of isolated nodes into the same cluster (respecting the maximum cluster weight limit).
  - cluster-always:         Pack any number of isolated nodes into the same cluster (respecting the maximum cluster weight limit).
  - match-during-two-hop:   Only match isolated nodes after two-hop clustering was triggered.
  - cluster-during-two-hop: Only cluster isolated nodes after two-hop clustering was triggered.
  )")
      ->capture_default_str();

  return lp;
}

CLI::Option_group *create_contraction_coarsening_options(CLI::App *app, Context &ctx) {
  auto *con = app->add_option_group("Coarsening -> Contraction");

  con->add_option("--c-contraction-algorithm", ctx.coarsening.contraction.algorithm)
      ->transform(CLI::CheckedTransformer(get_contraction_algorithms()).description(""))
      ->description(R"(Chooses the algorithm used for graph contraction:
  - buffered:   Use an edge buffer that is partially filled (slightly faster).
  - unbuffered: Use no edge buffer by remapping the coarse nodes (uses less memory).
  )")
      ->capture_default_str();
  con->add_option(
         "--c-contraction-unbuffered-impl", ctx.coarsening.contraction.unbuffered_implementation
  )
      ->transform(CLI::CheckedTransformer(get_contraction_implementations()).description(""))
      ->description(
          R"(Chooses implementation used for the unbuffered contraction:
  - single-phase:        Use single-phase unbuffered contraction.
  - two-phase:           Use two-phase unbuffered contraction.
  - growing-hash-tables: Use single-phase unbuffered contraction with growing hash tables.
  )"
      )
      ->capture_default_str();

  con->add_option(
         "--c-con-edge-buffer-fill-fraction",
         ctx.coarsening.contraction.edge_buffer_fill_fraction,
         "The fraction of the total edges with which to fill the edge buffer."
  )
      ->capture_default_str();

  return con;
}

CLI::Option_group *create_initial_partitioning_options(CLI::App *app, Context &ctx) {
  auto *ip = app->add_option_group("Initial Partitioning");

  ip->add_flag("--i-adaptive-epsilon", ctx.initial_partitioning.use_adaptive_epsilon)
      ->capture_default_str();

  // Pool
  ip->add_option("--i-p-min-num-repetitions", ctx.initial_partitioning.pool.min_num_repetitions)
      ->capture_default_str();
  ip->add_option("--i-p-max-num-repetitions", ctx.initial_partitioning.pool.max_num_repetitions)
      ->capture_default_str();

  // Refinement
  ip->add_flag("--i-r-disable", ctx.initial_partitioning.refinement.disabled)
      ->capture_default_str();
  ip->add_option("--i-r-num-iterations", ctx.initial_partitioning.refinement.num_iterations)
      ->capture_default_str();

  return ip;
}

CLI::Option_group *create_refinement_options(CLI::App *app, Context &ctx) {
  auto *refinement = app->add_option_group("Refinement");

  refinement->add_option("--r-algorithms", ctx.refinement.algorithms)
      ->transform(CLI::CheckedTransformer(get_kway_refinement_algorithms()).description(""))
      ->description(
          R"(Chooses the k-way refinement algorithm(s): expects a list of algorithms which are run in sequence. Algorithms can appear multiple times in the list.
The following algorithms can be used:
  - noop:               No k-way refinement.
  - lp:                 Label propagation (also see --r-lp-*).
  - fm:                 FM (also see --r-fm-*).
  - overload-balancer:  Rebalancer for maximum block weights.
  - underload-balancer: Rebalancer for minimum block weights.)"
      )
      ->capture_default_str();

  create_lp_refinement_options(app, ctx);
  create_kway_fm_refinement_options(app, ctx);
  create_jet_refinement_options(app, ctx);
  create_mtkahypar_refinement_options(app, ctx);

  return refinement;
}

CLI::Option_group *create_lp_refinement_options(CLI::App *app, Context &ctx) {
  auto *lp = app->add_option_group("Refinement -> Label Propagation");

  lp->add_option(
        "--r-lp-num-iterations",
        ctx.refinement.lp.num_iterations,
        "Number of label propagation iterations to perform."
  )
      ->capture_default_str();

  lp->add_option(
        "--r-lp-active-large-degree-threshold",
        ctx.refinement.lp.large_degree_threshold,
        "Ignore nodes that have a degree larger than this threshold."
  )
      ->capture_default_str();

  lp->add_option(
        "--r-lp-max-num-neighbors",
        ctx.refinement.lp.max_num_neighbors,
        "Maximum number of neighbors to consider for each nod.e"
  )
      ->capture_default_str();

  lp->add_option("--r-lp-impl", ctx.refinement.lp.impl)
      ->transform(CLI::CheckedTransformer(get_lp_implementations()).description(""))
      ->description(
          R"(Chooses the label propagation implementation:
  - single-phase:        Uses single-phase label propagation.
  - two-phase:           Uses two-phase label propagation.
  - growing-hash-tables: Uses single-phase label propagation with growing hash tables.
  )"
      )
      ->capture_default_str();

  lp->add_option("--r-lp-tie-breaking-strategy", ctx.refinement.lp.tie_breaking_strategy)
      ->transform(CLI::CheckedTransformer(get_tie_breaking_strategies()).description(""))
      ->description(
          R"(Chooses the tie breaking strategy:
  - geometric: Prefer nodes with same rating located at the end of a neighborhood.
  - uniform:   Select nodes with same rating uniformly at random.
  )"
      )
      ->capture_default_str();

  return lp;
}

CLI::Option_group *create_kway_fm_refinement_options(CLI::App *app, Context &ctx) {
  auto *fm = app->add_option_group("Refinement -> k-way FM");

  fm->add_option(
        "--r-fm-num-iterations",
        ctx.refinement.kway_fm.num_iterations,
        "Number of FM iterations to perform (higher = stronger, but slower)."
  )
      ->capture_default_str();
  fm->add_option(
        "--r-fm-num-seed-nodes",
        ctx.refinement.kway_fm.num_seed_nodes,
        "Number of seed nodes used to initialize a single localized search (lower = stronger, but "
        "slower)."
  )
      ->capture_default_str();
  fm->add_option(
        "--r-fm-abortion-threshold",
        ctx.refinement.kway_fm.abortion_threshold,
        "Stop FM iterations if the edge cut reduction of the previous "
        "iteration falls below this threshold (lower = weaker, but faster)."
  )
      ->capture_default_str();

  fm->add_flag(
      "--r-fm-lock-locally-moved-nodes{false},--r-fm-unlock-locally-moved-nodes",
      ctx.refinement.kway_fm.unlock_locally_moved_nodes,
      "If set, unlock all nodes after a batch that were only moved thread-locally, but not "
      "globally."
  );
  fm->add_flag(
      "--r-fm-lock-seed-nodes{false},--r-fm-unlock-seed-nodes",
      ctx.refinement.kway_fm.unlock_seed_nodes,
      "If set, keep seed nodes locked even if they were never moved. If this flag is not set, they "
      "are treated the same way as touched nodes."
  );

  // Flags for gain caches
  fm->add_option("--r-fm-gc", ctx.refinement.kway_fm.gain_cache_strategy)
      ->transform(CLI::CheckedTransformer(get_gain_cache_strategies()).description(""))
      ->capture_default_str();
  fm->add_option(
        "--r-fm-gc-const-hd-threshold",
        ctx.refinement.kway_fm.constant_high_degree_threshold,
        "If the selected gain cache strategy distinguishes between low- and high-degree nodes: use "
        "this as a constant threshold for high-degree nodes."
  )
      ->capture_default_str();
  fm->add_option(
        "--r-fm-gc-k-based-hd-threshold",
        ctx.refinement.kway_fm.k_based_high_degree_threshold,
        "If the selected gain cache strategy distinguishes between low- and high-degree nodes: use "
        "this multiplier times -k as a threshold for high-degree nodes."
  )
      ->capture_default_str();

  fm->add_option(
        "--r-fm-minimal-parallelism-pm",
        ctx.refinement.kway_fm.minimal_parallelism,
        "Fraction of cores that may not be idling before aborting the search."
  )
      ->capture_default_str();

  return fm;
}

CLI::Option_group *create_jet_refinement_options(CLI::App *app, Context &ctx) {
  auto *jet = app->add_option_group("Refinement -> Jet");

  jet->add_option("--r-jet-num-iterations", ctx.refinement.jet.num_iterations)
      ->capture_default_str();
  jet->add_option("--r-jet-num-fruitless-iterations", ctx.refinement.jet.num_fruitless_iterations)
      ->capture_default_str();
  jet->add_option("--r-jet-fruitless-threshold", ctx.refinement.jet.fruitless_threshold)
      ->capture_default_str();
  jet->add_option("--r-jet-num-rounds-on-fine-level", ctx.refinement.jet.num_rounds_on_fine_level)
      ->capture_default_str();
  jet->add_option(
         "--r-jet-num-rounds-on-coarse-level", ctx.refinement.jet.num_rounds_on_coarse_level
  )
      ->capture_default_str();
  jet->add_option(
         "--r-jet-initial-gain-temp-on-fine-level",
         ctx.refinement.jet.initial_gain_temp_on_fine_level
  )
      ->capture_default_str();
  jet->add_option(
         "--r-jet-final-gain-temp-on-fine-level", ctx.refinement.jet.final_gain_temp_on_fine_level
  )
      ->capture_default_str();
  jet->add_option(
         "--r-jet-initial-gain-temp-on-coarse-level",
         ctx.refinement.jet.initial_gain_temp_on_coarse_level
  )
      ->capture_default_str();
  jet->add_option(
         "--r-jet-final-gain-temp-on-coarse-level",
         ctx.refinement.jet.final_gain_temp_on_coarse_level
  )
      ->capture_default_str();

  return jet;
}

CLI::Option_group *create_mtkahypar_refinement_options(CLI::App *app, Context &ctx) {
  auto *mtkahypar = app->add_option_group("Refinement -> Mt-KaHyPar");

  mtkahypar
      ->add_option(
          "--r-mtkahypar-config",
          ctx.refinement.mtkahypar.config_filename,
          "Path to the Mt-KaHyPar configuration file."
      )
      ->capture_default_str();
  mtkahypar
      ->add_option(
          "--r-mtkahypar-config-fine",
          ctx.refinement.mtkahypar.fine_config_filename,
          "Path to the Mt-KaHyPar configuration file: this configuration is only used when "
          "refinement the input graph. Takes precedence over --r-mtkahypar-config."
      )
      ->capture_default_str();
  mtkahypar
      ->add_option(
          "--r-mtkahypar-config-coarse",
          ctx.refinement.mtkahypar.coarse_config_filename,
          "Path to the Mt-KaHyPar configuration file: this configuration is only used when "
          "refining coarse graphs. Takes precedence over --r-mtkahypar-config."
      )
      ->capture_default_str();

  return mtkahypar;
}

CLI::Option_group *create_debug_options(CLI::App *app, Context &ctx) {
  auto *debug = app->add_option_group("Debug");

  debug->add_option("--d-dump-graph-filename", ctx.debug.dump_graph_filename)
      ->capture_default_str();
  debug->add_option("--d-dump-partition-filename", ctx.debug.dump_partition_filename)
      ->capture_default_str();

  debug
      ->add_flag(
          "--d-dump-toplevel-graph",
          ctx.debug.dump_toplevel_graph,
          "Write the toplevel graph to disk. Note that this graph might be different from the "
          "input graph, as isolated nodes might have been removed and nodes might have been "
          "reordered."
      )
      ->capture_default_str();

  debug
      ->add_flag(
          "--d-dump-toplevel-partition",
          ctx.debug.dump_toplevel_partition,
          "Write the partition of the toplevel graph before- and after running refinement to disk. "
          "This partition should only be used together with the toplevel graph obtained via "
          "--d-dump-toplevel-graph."
      )
      ->capture_default_str();
  debug
      ->add_flag(
          "--d-dump-coarsest-graph",
          ctx.debug.dump_coarsest_graph,
          "Write the coarsest graph to disk. Note that the definition of "
          "'coarsest' depends on the partitioning scheme."
      )
      ->capture_default_str();

  debug
      ->add_flag(
          "--d-dump-coarsest-partition",
          ctx.debug.dump_coarsest_partition,
          "Write partition of the coarsest graph to disk. Note that the "
          "definition of 'coarsest' depends on the partitioning scheme."
      )
      ->capture_default_str();

  debug
      ->add_flag(
          "--d-dump-graph-hierarchy",
          ctx.debug.dump_graph_hierarchy,
          "Write the entire graph hierarchy to disk."
      )
      ->capture_default_str();

  debug
      ->add_flag(
          "--d-dump-partition-hierarchy",
          ctx.debug.dump_partition_hierarchy,
          "Write the entire partition hierarchy to disk."
      )
      ->capture_default_str();

  debug->add_flag(
      "--d-dump-everything",
      [&](auto) {
        ctx.debug.dump_toplevel_graph = true;
        ctx.debug.dump_toplevel_partition = true;
        ctx.debug.dump_coarsest_graph = true;
        ctx.debug.dump_coarsest_partition = true;
        ctx.debug.dump_graph_hierarchy = true;
        ctx.debug.dump_partition_hierarchy = true;
      },
      "Active all --d-dump-* options."
  );

  return debug;
}

} // namespace kaminpar::shm
