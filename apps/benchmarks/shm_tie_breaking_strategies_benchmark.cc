/*******************************************************************************
 * Tie-breaking strategies benchmarks for the shared-memory algorithm.
 *
 * @file:   shm_tie_breaking_strategies_benchmark.cc
 * @author: Daniel Salwasser
 * @date:   03.05.2024
 ******************************************************************************/
// clang-format off
#include <kaminpar-cli/kaminpar_arguments.h>
// clang-format on

#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/tie_breaking_strategies.h"

#include "kaminpar-common/console_io.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/random.h"
#include "kaminpar-common/timer.h"

using namespace kaminpar;
using namespace kaminpar::shm;

struct BenchmarkContext {
  int seed = 0;
  std::size_t num_iterations = 10000;
  std::size_t length = 25;
  bool output_histogram = false;
};

template <typename Strategy>
std::vector<std::size_t>
benchmark_strategy(int seed, std::size_t num_iterations, std::size_t length) {
  Strategy strategy;

  Random &random = Random::instance();
  random.reinit(seed);

  std::vector<std::size_t> histogram;
  histogram.resize(length);

  for (std::size_t i = 0; i < num_iterations; ++i) {
    strategy.init(0);

    for (std::size_t j = 1; j < length; ++j) {
      strategy.add(random, j);
    }

    const std::size_t selected = strategy.select(random);
    histogram[selected] += 1;
  }

  return std::move(histogram);
}

void print_histogram(const std::vector<std::size_t> &histogram) {
  LOG << "Histogram:";
  for (std::size_t i = 0; i < histogram.size(); ++i) {
    LOG << i << ": " << histogram[i];
  }
}

template <typename Strategy> void run_benchmark(BenchmarkContext ctx) {
  std::vector<std::size_t> histogram =
      benchmark_strategy<Strategy>(ctx.seed, ctx.num_iterations, ctx.length);

  if (ctx.output_histogram) {
    print_histogram(histogram);
  }
}

int main(int argc, char *argv[]) {
  BenchmarkContext ctx;

  CLI::App app("Shared-memory tie-breaking strategies benchmark");
  app.add_option("-s,--seed", ctx.seed, "Seed for random number generation.")
      ->default_val(ctx.seed);
  app.add_option("-i", ctx.num_iterations, "Number of iterations.")
      ->default_val(ctx.num_iterations);
  app.add_option("-l", ctx.length, "Length of each iteration.")->default_val(ctx.length);
  app.add_flag(
         "-p,--print-histogram",
         ctx.output_histogram,
         "Whether to print a histogram of the distributions."
  )
      ->default_val(ctx.output_histogram);
  CLI11_PARSE(app, argc, argv);

  GLOBAL_TIMER.reset();
  TIMED_SCOPE("Geometric series tie-breaking") {
    run_benchmark<GeometricTieBreakingStrategy<NodeID>>(ctx);
  };
  TIMED_SCOPE("Uniform tie-breaking (Naive)") {
    run_benchmark<NaiveUniformTieBreakingStrategy<NodeID>>(ctx);
  };
  TIMED_SCOPE("Uniform tie-breaking (Algorithm R)") {
    run_benchmark<UniformRTieBreakingStrategy<NodeID>>(ctx);
  };
  TIMED_SCOPE("Uniform tie-breaking (Algorithm L)") {
    run_benchmark<UniformLTieBreakingStrategy<NodeID>>(ctx);
  };
  STOP_TIMER();

  cio::print_delimiter("Input Summary");
  std::cout << "Seed:                         " << Random::get_seed() << "\n";

  cio::print_delimiter("Result Summary");
  Timer::global().print_human_readable(std::cout);

  return 0;
}
