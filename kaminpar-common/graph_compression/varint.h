/*******************************************************************************
 * Encoding and decoding methods for VarInts.
 *
 * @file:   varint.h
 * @author: Daniel Salwasser
 * @date:   11.11.2023
 ******************************************************************************/
#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

#ifdef KAMINPAR_COMPRESSION_FAST_DECODING
#include <immintrin.h>
#endif

#include "kaminpar-common/math.h"

namespace kaminpar {

/*!
 * Returns the maximum number of bytes that a VarInt needs to be stored.
 *
 * @tparam Int The type of integer whose encoded maximum length is returned.
 */
template <typename Int> [[nodiscard]] constexpr std::size_t varint_max_length() {
  return sizeof(Int) * 2;
}

/*!
 * Returns the number of bytes a VarInt needs to be stored.
 *
 * @tparam Int The type of integer whose encoded length is returned.
 * @param Int The integer to store.
 * @return The number of bytes the integer needs to be stored.
 */
template <typename Int> [[nodiscard]] std::size_t varint_length([[maybe_unused]] Int i) {
  return sizeof(Int) * 2;
}

/*!
 * Writes an integer to a memory location as a VarInt.
 *
 * @tparam Int The type of integer to encode.
 * @param Int The integer to store.
 * @param ptr The pointer to the memory location to write the integer to.
 * @return The number of bytes that the integer occupies at the memory location.
 */
template <typename Int> std::size_t varint_encode(Int i, std::uint8_t *ptr) {
  *reinterpret_cast<Int *>(ptr) = i;
  return sizeof(Int);
}

/*!
 * Writes an integer to a memory location as a VarInt.
 *
 * @tparam Int The type of integer to encode.
 * @param Int The integer to store.
 * @param ptr A pointer to the pointer to the memory location to write the integer to, which is
 * incremented accordingly.
 */
template <typename Int> void varint_encode(Int i, std::uint8_t **ptr) {
  *reinterpret_cast<Int *>(*ptr) = i;
  *ptr += sizeof(Int);
}

/*!
 * Reads an integer encoded as a VarInt from a memory location.
 *
 * @tparam Int The type of integer to decode.
 * @param ptr A pointer to the memory location to read the integer from.
 * @return The decoded integer.
 */
template <typename Int> [[nodiscard]] Int varint_decode(const std::uint8_t *data) {
  return *reinterpret_cast<const Int *>(data);
}

/*!
 * Reads an integer encoded as a VarInt from a memory location.
 *
 * @tparam Int The type of integer to decode.
 * @param ptr A pointer to the pointer to the memory location to read the integer from, which is
 * incremented accordingly.
 * @return The decoded integer.
 */
template <typename Int> [[nodiscard]] Int varint_decode_loop(const std::uint8_t **data) {
  const Int i = *reinterpret_cast<const Int *>(*data);
  *data += sizeof(Int);
  return i;
}

/*!
 * Reads an integer encoded as a VarInt from a memory location.
 *
 * @tparam Int The type of integer to decode.
 * @param ptr A pointer to the pointer to the memory location to read the integer from, which is
 * incremented accordingly.
 * @return The decoded integer.
 */
template <typename Int> [[nodiscard]] Int varint_decode_pext_unrolled(const std::uint8_t **data) {
  const Int i = *reinterpret_cast<const Int *>(*data);
  *data += sizeof(Int);
  return i;
}

/*!
 * Reads an integer encoded as a VarInt from a memory location.
 *
 * @tparam Int The type of integer to decode.
 * @param ptr A pointer to the pointer to the memory location to read the integer from, which is
 * incremented accordingly.
 * @return The decoded integer.
 */
template <typename Int> [[nodiscard]] Int varint_decode_pext_branchless(const std::uint8_t **data) {
  const Int i = *reinterpret_cast<const Int *>(*data);
  *data += sizeof(Int);
  return i;
}

/*!
 * Reads an integer encoded as a VarInt from a memory location.
 *
 * @tparam Int The type of integer to decode.
 * @param ptr A pointer to the pointer to the memory location to read the integer from, which is
 * incremented accordingly.
 * @return The decoded integer.
 */
template <typename Int> [[nodiscard]] Int varint_decode(const std::uint8_t **data) {
  const Int i = *reinterpret_cast<const Int *>(*data);
  *data += sizeof(Int);
  return i;
}

/*!
 * Returns the number of bytes a marked VarInt needs to be stored.
 *
 * @tparam Int The type of integer whose encoded length is returned.
 * @param Int The integer to store.
 * @return The number of bytes the integer needs to be stored.
 */
template <typename Int> [[nodiscard]] std::size_t marked_varint_length([[maybe_unused]] Int i) {
  return sizeof(Int) * 2;
}

/*!
 * Writes an integer to a memory location as a marked VarInt.
 *
 * @tparam Int The type of integer to encode.
 * @param Int The integer to store.
 * @param marker_set Whether the integer is marked.
 * @param ptr The pointer to the memory location to write the integer to.
 */
template <typename Int> std::size_t marked_varint_encode(Int i, bool marked, std::uint8_t *ptr) {
  if (marked) {
    i |= math::kSetMSB<Int>;
  }

  *reinterpret_cast<Int *>(ptr) = i;
  return sizeof(Int);
}

/*!
 * Writes an integer to a memory location as a marked VarInt.
 *
 * @tparam Int The type of integer to encode.
 * @param Int The integer to store.
 * @param marker_set Whether the integer is marked.
 * @param ptr The pointer to the memory location to write the integer to.
 */
template <typename Int> void marked_varint_encode(Int i, const bool marked, std::uint8_t **ptr) {
  if (marked) {
    i |= math::kSetMSB<Int>;
  }

  *reinterpret_cast<Int *>(*ptr) = i;
  *ptr += sizeof(Int);
}

/*!
 * Reads an integer encoded as a marked VarInt from a memory location.
 *
 * @tparam Int The type of integer to decode.
 * @param ptr A pointer to the memory location to read the integer from.
 * @return A pair consisting of the decoded integer and whether the marker is set.
 */
template <typename Int>
[[nodiscard]] std::pair<Int, bool> marked_varint_decode(const std::uint8_t *ptr) {
  Int i = *reinterpret_cast<const Int *>(ptr);
  const bool is_marked = math::is_msb_set(i);
  i &= ~math::kSetMSB<Int>;
  return std::make_pair(i, is_marked);
}

/*!
 * Reads an integer encoded as a marked VarInt from a memory location.
 *
 * @tparam Int The type of integer to decode.
 * @param ptr A pointer to the pointer to the memory location to read the integer from, which is
 * incremented accordingly.
 * @return A pair consisting of the decoded integer and whether the markes is set.
 */
template <typename Int>
[[nodiscard]] std::pair<Int, bool> marked_varint_decode(const std::uint8_t **ptr) {
  Int i = *reinterpret_cast<const Int *>(*ptr);
  const bool is_marked = math::is_msb_set(i);
  i &= ~math::kSetMSB<Int>;
  *ptr += sizeof(Int);
  return std::make_pair(i, is_marked);
}

/*!
 * Encodes a signed integer using zigzag encoding.
 *
 * @param i The signed integer to encode.
 * @return The encoded integer.
 */
template <typename Int> [[nodiscard]] std::make_unsigned_t<Int> zigzag_encode(const Int i) {
  return (i >> (sizeof(Int) * 8 - 1)) ^ (i << 1);
}

/*!
 * Decodes a zigzag encoded integer.
 *
 * @param i The zigzag encoded integer to decode.
 * @return The decoded integer.
 */
template <typename Int> [[nodiscard]] std::make_signed_t<Int> zigzag_decode(const Int i) {
  return (i >> 1) ^ -(i & 1);
}

/*!
 * Returns the number of bytes a signed VarInt needs to be stored.
 *
 * @tparam Int The type of integer whose encoded length is returned.
 * @param Int The integer to store.
 * @return The number of bytes the integer needs to be stored.
 */
template <typename Int> [[nodiscard]] std::size_t signed_varint_length(const Int i) {
  return varint_length(zigzag_encode(i));
}

/*!
 * Writes an integer to a memory location as a signed VarInt.
 *
 * @tparam Int The type of integer to encode.
 * @param Int The integer to store.
 * @param ptr The pointer to the memory location to write the integer to.
 * @return The number of bytes that the integer occupies at the memory location.
 */
template <typename Int> std::size_t signed_varint_encode(const Int i, std::uint8_t *ptr) {
  return varint_encode(zigzag_encode(i), ptr);
}

/*!
 * Writes an integer to a memory location as a signed VarInt.
 *
 * @tparam Int The type of integer to encode.
 * @param Int The integer to store.
 * @param ptr A pointer to the pointer to the memory location to write the integer to, which is
 * incremented accordingly.
 */
template <typename Int> void signed_varint_encode(const Int i, std::uint8_t **ptr) {
  varint_encode(zigzag_encode(i), ptr);
}

/*!
 * Reads an integer encoded as a signed VarInt from a memory location.
 *
 * @tparam Int The type of integer to decode.
 * @param ptr A pointer to the memory location to read the integer from.
 * @return The decoded integer.
 */
template <typename Int> [[nodiscard]] Int signed_varint_decode(const std::uint8_t *data) {
  return zigzag_decode(varint_decode<std::make_unsigned_t<Int>>(data));
}

/*!
 * Reads an integer encoded as a signed VarInt from a memory location.
 *
 * @tparam Int The type of integer to decode.
 * @param ptr A pointer to the pointer to the memory location to read the integer from, which is
 * incremented accordingly.
 * @return The decoded integer.
 */
template <typename Int> [[nodiscard]] Int signed_varint_decode(const std::uint8_t **data) {
  return zigzag_decode(varint_decode<std::make_unsigned_t<Int>>(data));
}

} // namespace kaminpar
