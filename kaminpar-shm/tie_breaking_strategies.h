/******************************************************************************
 * Tie-breaking strategies for selecting a best cluster during label propagation.
 *
 * @file:   tie_breaking_strategies.h
 * @author: Daniel Salwasser
 * @date:   03.05.2022
 ******************************************************************************/
#pragma once

#include <cmath>
#include <vector>

#include "kaminpar-common/random.h"

namespace kaminpar::shm {

/*!
 * Tie-breaking strategy that favors nodes considered last.
 */
template <typename ID> class GeometricTieBreakingStrategy {
public:
  void init(ID i) {
    _selected = i;
  }

  void add(Random &random, ID i) {
    if (random.random_bool()) {
      _selected = i;
    }
  }

  [[nodiscard]] ID select(Random &) {
    return _selected;
  }

private:
  ID _selected;
};

/*!
 * Tie-breaking strategy that selects nodes uniformly at random. It implements Algorithm R, which
 * uses n random numbers where n is the number of nodes considered.
 */
template <typename ID> class NaiveUniformTieBreakingStrategy {
public:
  void init(ID i) {
    _entries.clear();
    _entries.push_back(i);
  }

  void add(Random &, ID i) {
    _entries.push_back(i);
  }

  [[nodiscard]] ID select(Random &random) {
    const ID index = random.random_index(0, _entries.size());
    return _entries[index];
  }

private:
  std::vector<ID> _entries;
};

/*!
 * Tie-breaking strategy that selects nodes uniformly at random. It implements Algorithm R, which
 * uses n random numbers where n is the number of nodes considered.
 */
template <typename ID> class UniformRTieBreakingStrategy {
public:
  void init(ID i) {
    _selected = i;
    _counter = 1;
  }

  void add(Random &random, ID i) {
    const ID index = random.random_index(0, ++_counter);
    if (index == 0) {
      _selected = i;
    }
  }

  [[nodiscard]] ID select(Random &) {
    return _selected;
  }

private:
  ID _selected;
  ID _counter;
};

/*!
 * Tie-breaking strategy that selects nodes uniformly at random. It implements Algorithm L, which
 * uses log(n) random numbers where n is the number of nodes considered.
 */
template <typename ID> class UniformLTieBreakingStrategy {
public:
  void init(ID i) {
    _selected = i;
    _counter = 0;
    _next = 0;
  }

  void add(Random &random, ID i) {
    if (_counter == 0) {
      _w = random.random_double();
      compute_next(random);
    }

    if (++_counter == _next) {
      _selected = i;
      compute_next(random);
    }
  }

  [[nodiscard]] ID select(Random &) {
    return _selected;
  }

private:
  ID _selected;
  ID _counter;
  ID _next;
  double _w;

  inline void compute_next(Random &random) {
    _next += static_cast<ID>(std::log(random.random_double()) / std::log(1 - _w)) + 1;
    _w *= random.random_double();
  }
};

} // namespace kaminpar::shm
