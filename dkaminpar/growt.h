/*******************************************************************************
 * @file:   growt.h
 *
 * @author: Daniel Seemaier
 * @date:   30.09.21
 * @brief:  Include growt and suppress -Wpedantic warnings.
 ******************************************************************************/
#pragma once

#include <type_traits>

#include <tbb/enumerable_thread_specific.h>

#include "dkaminpar/definitions.h"

#pragma GCC diagnostic   push
#pragma clang diagnostic push
#pragma GCC diagnostic   ignored "-Wpedantic"
#pragma clang diagnostic ignored "-Wpedantic"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wuninitialized"
#include <allocator/alignedallocator.hpp>
#include <data-structures/table_config.hpp>
#include <utils/hash/murmur2_hash.hpp>
#pragma GCC diagnostic   pop
#pragma clang diagnostic pop

namespace dkaminpar::growt {
using DefaultHasherType    = utils_tm::hash_tm::murmur2_hash;
using DefaultAllocatorType = ::growt::AlignedAllocator<>;

namespace internal {
// workaround 32 bit value bug in growt
template <typename Type>
using Ensure64BitType = std::conditional_t<std::numeric_limits<Type>::is_signed, GlobalNodeWeight, GlobalNodeID>;
} // namespace internal

template <typename Value>
using GlobalNodeIDMap = typename ::growt::table_config<
    GlobalNodeID, internal::Ensure64BitType<Value>, DefaultHasherType, DefaultAllocatorType, hmod::growable,
    hmod::deletion>::table_type;

using StaticGhostNodeMapping =
    typename ::growt::table_config<GlobalNodeID, GlobalNodeID, DefaultHasherType, DefaultAllocatorType>::table_type;

template <typename Map>
auto create_handle_ets(Map& map) {
    return tbb::enumerable_thread_specific<growt::GlobalNodeIDMap<NodeID>::handle_type>{[&] {
        return map.get_handle();
    }};
}
} // namespace dkaminpar::growt
