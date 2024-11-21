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
  return (sizeof(Int) * 8) / 7 + 1;
}

/*!
 * Returns the number of bytes a VarInt needs to be stored.
 *
 * @tparam Int The type of integer whose encoded length is returned.
 * @param Int The integer to store.
 * @return The number of bytes the integer needs to be stored.
 */
template <typename Int> [[nodiscard]] std::size_t varint_length(Int i) {
  const bool single_byte = (i & ~static_cast<Int>(0b11111)) == 0;
  if (single_byte) {
    return 1;
  }

  return 1 + math::div_ceil(std::bit_width(i >> 5), 8);
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
  const std::size_t length = varint_length(i);
  const std::uint64_t vbyte = (i << 3) | (length - 1);

  *reinterpret_cast<std::uint64_t *>(ptr) = vbyte;
  return length;
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
  const std::size_t length = varint_length(i);
  const std::uint64_t vbyte = (i << 3) | (length - 1);

  *reinterpret_cast<std::uint64_t *>(*ptr) = vbyte;
  *ptr += length;
}

/*!
 * Reads an integer encoded as a VarInt from a memory location.
 *
 * @tparam Int The type of integer to decode.
 * @param ptr A pointer to the memory location to read the integer from.
 * @return The decoded integer.
 */
template <typename Int> [[nodiscard]] Int varint_decode(const std::uint8_t *data) {
  const std::uint64_t vbyte = *reinterpret_cast<const std::uint64_t *>(data);
  const std::uint64_t header_bits = vbyte & 0b111;

  static constexpr std::uint64_t kLookupTable[] = {
      0xFF, 0xFFFF, 0xFFFFFF, 0xFFFFFFFF, 0xFFFFFFFFFF
  };
  const std::uint64_t mask = kLookupTable[header_bits];

  return (vbyte & mask) >> 3;
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
  const std::uint64_t vbyte = *reinterpret_cast<const std::uint64_t *>(*data);
  const std::uint64_t header_bits = vbyte & 0b111;

  static constexpr std::uint64_t kLookupTable[] = {
      0xFF, 0xFFFF, 0xFFFFFF, 0xFFFFFFFF, 0xFFFFFFFFFF
  };
  const std::uint64_t mask = kLookupTable[header_bits];

  const std::uint64_t length = header_bits + 1;
  *data += length;

  return (vbyte & mask) >> 3;
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
#ifdef KAMINPAR_COMPRESSION_FAST_DECODING
  if constexpr (sizeof(Int) == 4) {
    const std::uint8_t *data_ptr = *data;
    if ((data_ptr[0] & 0b10000000) == 0) {
      const std::uint32_t result = *data_ptr & 0b01111111;
      *data += 1;
      return result;
    }

    if ((data_ptr[1] & 0b10000000) == 0) {
      const std::uint32_t result =
          _pext_u32(*reinterpret_cast<const std::uint32_t *>(data_ptr), 0x7F7F);
      *data += 2;
      return result;
    }

    if ((data_ptr[2] & 0b10000000) == 0) {
      const std::uint32_t result =
          _pext_u32(*reinterpret_cast<const std::uint32_t *>(data_ptr), 0x7F7F7F);
      *data += 3;
      return result;
    }

    if ((data_ptr[3] & 0b10000000) == 0) {
      const std::uint32_t result =
          _pext_u32(*reinterpret_cast<const std::uint32_t *>(data_ptr), 0x7F7F7F7F);
      *data += 4;
      return result;
    }

    const std::uint32_t result = static_cast<std::uint32_t>(
        _pext_u64(*reinterpret_cast<const std::uint64_t *>(data_ptr), 0x7F7F7F7F7F)
    );
    *data += 5;
    return result;
  } else if constexpr (sizeof(Int) == 8) {
    const std::uint8_t *data_ptr = *data;
    if ((data_ptr[0] & 0b10000000) == 0) {
      const std::uint64_t result = *data_ptr & 0b01111111;
      *data += 1;
      return result;
    }

    if ((data_ptr[1] & 0b10000000) == 0) {
      const std::uint64_t result =
          _pext_u32(*reinterpret_cast<const std::uint32_t *>(data_ptr), 0x7F7F);
      *data += 2;
      return result;
    }

    if ((data_ptr[2] & 0b10000000) == 0) {
      const std::uint64_t result =
          _pext_u32(*reinterpret_cast<const std::uint32_t *>(data_ptr), 0x7F7F7F);
      *data += 3;
      return result;
    }

    if ((data_ptr[3] & 0b10000000) == 0) {
      const std::uint64_t result =
          _pext_u32(*reinterpret_cast<const std::uint32_t *>(data_ptr), 0x7F7F7F7F);
      *data += 4;
      return result;
    }

    if ((data_ptr[4] & 0b10000000) == 0) {
      const std::uint64_t result =
          _pext_u64(*reinterpret_cast<const std::uint64_t *>(data_ptr), 0x7F7F7F7F7F);
      *data += 5;
      return result;
    }

    if ((data_ptr[5] & 0b10000000) == 0) {
      const std::uint64_t result =
          _pext_u64(*reinterpret_cast<const std::uint64_t *>(data_ptr), 0x7F7F7F7F7F7F);
      *data += 6;
      return result;
    }

    if ((data_ptr[6] & 0b10000000) == 0) {
      const std::uint64_t result =
          _pext_u64(*reinterpret_cast<const std::uint64_t *>(data_ptr), 0x7F7F7F7F7F7F7F);
      *data += 7;
      return result;
    }

    if ((data_ptr[7] & 0b10000000) == 0) {
      const std::uint64_t result =
          _pext_u64(*reinterpret_cast<const std::uint64_t *>(data_ptr), 0x7F7F7F7F7F7F7F7F);
      *data += 8;
      return result;
    }

    if ((data_ptr[8] & 0b10000000) == 0) {
      const std::uint64_t result =
          _pext_u64(*reinterpret_cast<const std::uint64_t *>(data_ptr), 0x7F7F7F7F7F7F7F7F) |
          (static_cast<std::uint64_t>(data_ptr[8] & 0b01111111) << 56);
      *data += 9;
      return result;
    }

    const std::uint64_t result =
        _pext_u64(*reinterpret_cast<const std::uint64_t *>(data_ptr), 0x7F7F7F7F7F7F7F7F) |
        (static_cast<std::uint64_t>(data_ptr[8] & 0b01111111) << 56) |
        (static_cast<std::uint64_t>(data_ptr[9]) << 63);
    *data += 10;
    return result;
  }
#else
  return varint_decode_loop<Int>(data);
#endif
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
#ifdef KAMINPAR_COMPRESSION_FAST_DECODING
  if constexpr (sizeof(Int) == 4) {
    const std::uint8_t *data_ptr = *data;

    const std::uint64_t word = *reinterpret_cast<const std::uint64_t *>(data_ptr);
    const std::uint64_t continuation_bits = ~word & 0x8080808080;
    const std::uint64_t mask = continuation_bits ^ (continuation_bits - 1);
    const std::uint64_t length = (std::countr_zero(continuation_bits) + 1) / 8;

    const Int result = _pext_u64(word & mask, 0x7F7F7F7F7F);
    *data += length;
    return result;
  }
#else
  return varint_decode_loop<Int>(data);
#endif
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
  return varint_decode_pext_unrolled<Int>(data);
}

/*!
 * Returns the number of bytes a marked VarInt needs to be stored.
 *
 * @tparam Int The type of integer whose encoded length is returned.
 * @param Int The integer to store.
 * @return The number of bytes the integer needs to be stored.
 */
template <typename Int> [[nodiscard]] std::size_t marked_varint_length(Int i) {
  const bool single_byte = (i & ~static_cast<Int>(0b1111)) == 0;
  if (single_byte) {
    return 1;
  }

  return 1 + math::div_ceil(std::bit_width(i >> 4), 8);
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
  const std::size_t length = marked_varint_length(i);
  const std::uint64_t vbyte = (i << 4) | ((length - 1) << 1) | (marked ? 1 : 0);

  *reinterpret_cast<std::uint64_t *>(ptr) = vbyte;
  return length;
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
  const std::size_t length = marked_varint_length(i);
  const std::uint64_t vbyte = (i << 4) | ((length - 1) << 1) | (marked ? 1 : 0);

  *reinterpret_cast<std::uint64_t *>(*ptr) = vbyte;
  *ptr += length;
}

/*!
 * Reads an integer encoded as a marked VarInt from a memory location.
 *
 * @tparam Int The type of integer to decode.
 * @param ptr A pointer to the memory location to read the integer from.
 * @return A pair consisting of the decoded integer and whether the marker is set.
 */
template <typename Int>
[[nodiscard]] std::pair<Int, bool> marked_varint_decode(const std::uint8_t *data) {
  const std::uint64_t vbyte = *reinterpret_cast<const std::uint64_t *>(data);
  const std::uint64_t header_bits = (vbyte >> 1) & 0b111;

  static constexpr std::uint64_t kLookupTable[] = {
      0xFF, 0xFFFF, 0xFFFFFF, 0xFFFFFFFF, 0xFFFFFFFFFF
  };
  const std::uint64_t mask = kLookupTable[header_bits];

  return std::make_pair((vbyte & mask) >> 4, (vbyte & 1) != 0);
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
[[nodiscard]] std::pair<Int, bool> marked_varint_decode(const std::uint8_t **data) {
  const std::uint64_t vbyte = *reinterpret_cast<const std::uint64_t *>(*data);
  const std::uint64_t header_bits = (vbyte >> 1) & 0b111;

  static constexpr std::uint64_t kLookupTable[] = {
      0xFF, 0xFFFF, 0xFFFFFF, 0xFFFFFFFF, 0xFFFFFFFFFF
  };
  const std::uint64_t mask = kLookupTable[header_bits];

  const std::uint64_t length = header_bits + 1;
  *data += length;

  return std::make_pair((vbyte & mask) >> 4, (vbyte & 1) != 0);
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
