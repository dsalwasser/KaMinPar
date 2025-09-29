/*******************************************************************************
 * General-purpose sorting algorithm interface.
 *
 * @file:   sort.h
 * @author: Daniel Salwasser
 * @date:   29.09.2025
 ******************************************************************************/
#pragma once

#include <functional>

#ifdef KAMINPAR_IPS4O_FOUND
#include <ips4o/ips4o.hpp>
#else
#include <algorithm>

#include <tbb/parallel_sort.h>
#endif

namespace kaminpar::sorting {

template <typename Iterator, typename Comp = std::less<>>
void sort(Iterator begin, Iterator end, Comp comp = Comp()) {
#ifdef KAMINPAR_IPS4O_FOUND
  ips4o::sort(begin, end, comp);
#else
  std::sort(begin, end, comp);
#endif
}

template <typename Iterator, typename Comp = std::less<>>
void parallel_sort(Iterator begin, Iterator end, Comp comp = Comp()) {
#ifdef KAMINPAR_IPS4O_FOUND
  ips4o::parallel::sort(begin, end, comp);
#else
  tbb::parallel_sort(begin, end, comp);
#endif
}

} // namespace kaminpar::sorting
