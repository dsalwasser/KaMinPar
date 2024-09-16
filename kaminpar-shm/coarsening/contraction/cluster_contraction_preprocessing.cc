/*******************************************************************************
 * Common preprocessing utilities for cluster contraction implementations.
 *
 * @file:   cluster_contraction_preprocessing.cc
 * @author: Daniel Seemaier
 * @author: Daniel Salwasser
 * @date:   21.09.2021
 ******************************************************************************/
#include "kaminpar-shm/coarsening/contraction/cluster_contraction_preprocessing.h"

#include <utility>
#include <vector>

#include <tbb/cache_aligned_allocator.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/task_arena.h>

#include "kaminpar-common/degree_buckets.h"
#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/parallel/algorithm.h"
#include "kaminpar-common/parallel/loops.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm::contraction {

namespace {
SET_DEBUG(false);

template <typename Lambda>
[[nodiscard]] StaticArray<NodeID> sort_by_degree_buckets(const NodeID n, Lambda &&degree_bucket) {
  static_assert(std::is_invocable_r_v<NodeID, Lambda, NodeID>);

  RECORD("permutation") StaticArray<NodeID> permutation(n);

  const std::size_t num_threads = std::min<std::size_t>(tbb::this_task_arena::max_concurrency(), n);

  using Buckets = std::array<NodeID, kNumberOfDegreeBuckets<NodeID> + 1>;
  std::vector<Buckets, tbb::cache_aligned_allocator<Buckets>> local_buckets(num_threads + 1);
  parallel::deterministic_for<NodeID>(
      0,
      n,
      [&](const NodeID from, const NodeID to, const std::size_t thread_id) {
        for (NodeID u = from; u < to; ++u) {
          const NodeID bucket = degree_bucket(u);
          permutation[u] = local_buckets[thread_id + 1][bucket]++;
        }
      }
  );

  // Build a table of prefix numbers to correct the position of each node in the final permutation
  // After the previous loop, permutation[u] contains the position of u in the thread-local bucket.
  // (i)  account for smaller buckets --> add prefix computed in global_buckets
  // (ii) account for the same bucket in smaller processor IDs --> add prefix computed in
  //      local_buckets
  Buckets global_buckets{};
  for (std::size_t id = 1; id < num_threads + 1; ++id) {
    for (std::size_t i = 0; i + 1 < global_buckets.size(); ++i) {
      global_buckets[i + 1] += local_buckets[id][i];
    }
  }
  for (std::size_t i = 2; i < global_buckets.size(); ++i) {
    global_buckets[i] += global_buckets[i - 1];
  }
  for (std::size_t i = 0; i < global_buckets.size(); ++i) {
    for (std::size_t id = 0; id + 1 < num_threads; ++id) {
      local_buckets[id + 1][i] += local_buckets[id][i];
    }
  }

  // Apply offsets to obtain global permutation
  parallel::deterministic_for<NodeID>(
      0,
      n,
      [&](const NodeID from, const NodeID to, const std::size_t thread_id) {
        for (NodeID u = from; u < to; ++u) {
          const NodeID bucket = degree_bucket(u);
          permutation[u] += global_buckets[bucket] + local_buckets[thread_id][bucket];
        }
      }
  );

  return permutation;
}

} // namespace

std::tuple<bool, NodeID, StaticArray<NodeID>> contraction_preprocessing(
    const Context &ctx, const Graph &graph, StaticArray<NodeID> clustering, MemoryContext &m_ctx
) {
  auto &leader_mapping = m_ctx.leader_mapping;

  const NodeID n = graph.n();
  START_TIMER("Compute cluster mapping");
  TIMED_SCOPE("Allocation") {
    SCOPED_HEAP_PROFILER("Compute cluster mapping");

    if (leader_mapping.size() < n) {
      RECORD("leader_mapping") leader_mapping.resize(n, static_array::noinit);
      RECORD_LOCAL_DATA_STRUCT(leader_mapping, leader_mapping.size() * sizeof(NodeID));
    }
  };

  tbb::parallel_for(tbb::blocked_range<NodeID>(0, n), [&](const auto &r) {
    for (NodeID u = r.begin(); u != r.end(); ++u) {
      leader_mapping[u] = 0;
    }
  });
  tbb::parallel_for(tbb::blocked_range<NodeID>(0, n), [&](const auto &r) {
    for (NodeID u = r.begin(); u != r.end(); ++u) {
      __atomic_store_n(&leader_mapping[clustering[u]], 1, __ATOMIC_RELAXED);
    }
  });
  parallel::prefix_sum(leader_mapping.begin(), leader_mapping.begin() + n, leader_mapping.begin());

  if (ctx.coarsening.contraction.sort_by_deg_buckets) {
    const auto permutation = TIMED_SCOPE("Rearrange nodes by degree buckets") {
      SCOPED_HEAP_PROFILER("Rearrange nodes by degree buckets");

      StaticArray<NodeID> cluster_sizes(graph.n());
      tbb::parallel_for(tbb::blocked_range<NodeID>(0, n), [&](const auto &r) {
        for (NodeID u = r.begin(); u != r.end(); ++u) {
          const NodeID c_u = __atomic_load_n(&leader_mapping[clustering[u]], __ATOMIC_RELAXED) - 1;
          __atomic_fetch_add(&cluster_sizes[c_u], 1, __ATOMIC_RELAXED);
          clustering[u] = c_u;
        }
      });

      const NodeID c_n = leader_mapping[n - 1];
      return sort_by_degree_buckets(c_n, [&](const NodeID c_u) {
        const NodeID degree = cluster_sizes[c_u];
        return degree_bucket(degree);
      });
    };

    tbb::parallel_for(tbb::blocked_range<NodeID>(0, n), [&](const auto &r) {
      for (NodeID u = r.begin(); u != r.end(); ++u) {
        clustering[u] = permutation[clustering[u]];
      }
    });
  } else {
    tbb::parallel_for(tbb::blocked_range<NodeID>(0, n), [&](const auto &r) {
      for (NodeID u = r.begin(); u != r.end(); ++u) {
        const NodeID c_u = __atomic_load_n(&leader_mapping[clustering[u]], __ATOMIC_RELAXED) - 1;
        clustering[u] = c_u;
      }
    });
  }
  STOP_TIMER();

  auto &buckets = m_ctx.buckets;
  auto &buckets_index = m_ctx.buckets_index;

  NodeID c_n = leader_mapping[n - 1];
  NodeID desired_c_n = n / ctx.coarsening.clustering.shrink_factor;

  const double U = ctx.coarsening.clustering.forced_level_upper_factor;
  const double L = ctx.coarsening.clustering.forced_level_lower_factor;
  const NodeID C = ctx.coarsening.contraction_limit;
  const BlockID k = ctx.partition.k;
  const int p = ctx.parallel.num_threads;
  if (ctx.coarsening.clustering.forced_kc_level) {
    if (n > U * C * k) {
      desired_c_n = std::max<NodeID>(desired_c_n, L * C * k);
    }
  }
  if (ctx.coarsening.clustering.forced_pc_level) {
    if (n > U * C * p) {
      desired_c_n = std::max<NodeID>(desired_c_n, L * C * p);
    }
  }

  START_TIMER("Fill cluster buckets");
  TIMED_SCOPE("Allocation") {
    SCOPED_HEAP_PROFILER("Fill cluster buckets");

    if (buckets.size() < n) {
      RECORD("buckets") buckets.resize(n, static_array::noinit);
      RECORD_LOCAL_DATA_STRUCT(buckets, buckets.size() * sizeof(NodeID));
    }

    const auto alloc_buckets_index = [&](NodeID size) {
      if (buckets_index.size() < size) {
        RECORD("buckets_index") buckets_index.resize(size, static_array::noinit);
        RECORD_LOCAL_DATA_STRUCT(buckets_index, buckets_index.size() * sizeof(NodeID));
      }
    };
    if (c_n < desired_c_n) {
      alloc_buckets_index(desired_c_n + 1);
    } else {
      alloc_buckets_index(c_n + 1);
    }
  };

  tbb::parallel_for(tbb::blocked_range<NodeID>(0, c_n + 1), [&](const auto &r) {
    for (NodeID c_u = r.begin(); c_u != r.end(); ++c_u) {
      buckets_index[c_u] = 0;
    }
  });
  tbb::parallel_for(tbb::blocked_range<NodeID>(0, n), [&](const auto &r) {
    for (NodeID u = r.begin(); u != r.end(); ++u) {
      __atomic_fetch_add(&buckets_index[clustering[u]], 1, __ATOMIC_RELAXED);
    }
  });
  parallel::prefix_sum(
      buckets_index.begin(), buckets_index.begin() + c_n + 1, buckets_index.begin()
  );
  tbb::parallel_for(tbb::blocked_range<NodeID>(0, n), [&](const auto &r) {
    for (NodeID u = r.begin(); u != r.end(); ++u) {
      buckets[__atomic_sub_fetch(&buckets_index[clustering[u]], 1, __ATOMIC_RELAXED)] = u;
    }
  });
  STOP_TIMER();

  const bool force_level = c_n < desired_c_n;
  if (force_level) {
    SCOPED_TIMER("Force coarse level");
    DBG << "Increasing number of clusters from " << c_n << " to " << desired_c_n;

    NodeID last_c_n = desired_c_n;
    NodeID cur_c_n = c_n;
    for (NodeID i = 0; i < c_n; ++i) {
      const NodeID c_u = c_n - i - 1;

      const NodeID bucket_start = buckets_index[c_u];
      const NodeID bucket_end = buckets_index[c_u + 1];

      NodeID degree = bucket_end - bucket_start;

      cur_c_n += degree - 1;
      if (cur_c_n > desired_c_n) {
        degree -= cur_c_n - desired_c_n + 1;
      }

      for (NodeID j = 0; j < degree; ++j) {
        const NodeID new_c_u = last_c_n - j - 1;
        const NodeID new_bucket_start = bucket_end - j - 1;
        buckets_index[new_c_u] = new_bucket_start;
        clustering[buckets[new_bucket_start]] = new_c_u;
      }
      last_c_n -= degree;

      if (cur_c_n >= desired_c_n) {
        break;
      }
    }

    c_n = desired_c_n;
    buckets_index[c_n] = buckets_index[c_n - 1] + 1;
  }

  return std::make_tuple(force_level, c_n, std::move(clustering));
}

} // namespace kaminpar::shm::contraction
