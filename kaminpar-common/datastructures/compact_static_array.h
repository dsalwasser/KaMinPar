/*******************************************************************************
 * A static array which stores integers with only as many bytes as the largest
 * integer requires.
 *
 * @file:   compact_static_array.h
 * @author: Daniel Salwasser
 * @date:   12.01.2024
 ******************************************************************************/
#pragma once

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>

#include "kaminpar-common/assert.h"
#include "kaminpar-common/heap_profiler.h"

namespace kaminpar {

/*!
 * A static array which stores integers with only as many bytes as the largest integer requires.
 *
 * @tparam Int The type of integer to store.
 */
template <typename Int> class CompactStaticArray {
  static_assert(std::numeric_limits<Int>::is_integer);

  class CompactStaticArrayIterator {
  public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = Int;
    using reference = Int &;
    using pointer = Int *;
    using difference_type = std::ptrdiff_t;

    CompactStaticArrayIterator(
        const std::uint8_t byte_width, const Int mask, const std::uint8_t *data
    )
        : _byte_width(byte_width),
          _mask(mask),
          _data(data) {}

    CompactStaticArrayIterator(const CompactStaticArrayIterator &other) = default;
    CompactStaticArrayIterator &operator=(const CompactStaticArrayIterator &other) = default;

    Int operator*() const {
      return *reinterpret_cast<const Int *>(_data) & _mask;
    }

    pointer operator->() const {
      return *reinterpret_cast<const Int *>(_data) & _mask;
    }

    reference operator[](const difference_type n) const {
      return *reinterpret_cast<const Int *>(_data + _byte_width * n) & _mask;
    }

    CompactStaticArrayIterator &operator++() {
      return _data += _byte_width, *this;
    }

    CompactStaticArrayIterator &operator--() {
      return _data -= _byte_width, *this;
    }

    CompactStaticArrayIterator operator++(int) const {
      return CompactStaticArrayIterator{_byte_width, _mask, _data + _byte_width};
    }

    CompactStaticArrayIterator operator--(int) const {
      return CompactStaticArrayIterator{_byte_width, _mask, _data - _byte_width};
    }

    CompactStaticArrayIterator operator+(const difference_type n) const {
      return CompactStaticArrayIterator{_byte_width, _mask, _data + _byte_width * n};
    }

    CompactStaticArrayIterator operator-(const difference_type n) const {
      return CompactStaticArrayIterator{_byte_width, _mask, _data - _byte_width * n};
    }

    CompactStaticArrayIterator &operator+=(const difference_type n) {
      return _data += _byte_width * n, *this;
    }

    CompactStaticArrayIterator &operator-=(const difference_type n) {
      return _data -= _byte_width * n, *this;
    }

    difference_type operator+(const CompactStaticArrayIterator &other) const {
      return (reinterpret_cast<difference_type>(_data) / _byte_width) +
             (reinterpret_cast<difference_type>(other._data) / _byte_width);
    }

    difference_type operator-(const CompactStaticArrayIterator &other) const {
      return (reinterpret_cast<difference_type>(_data) / _byte_width) -
             (reinterpret_cast<difference_type>(other._data) / _byte_width);
    }

    bool operator==(const CompactStaticArrayIterator &other) const {
      return _data == other._data;
    }

    bool operator!=(const CompactStaticArrayIterator &other) const {
      return _data != other._data;
    }

    bool operator>(const CompactStaticArrayIterator &other) const {
      return _data > other._data;
    }

    bool operator<(const CompactStaticArrayIterator &other) const {
      return _data < other._data;
    }

    bool operator>=(const CompactStaticArrayIterator &other) const {
      return _data >= other._ptr;
    }

    bool operator<=(const CompactStaticArrayIterator &other) const {
      return _data <= other._data;
    }

  private:
    const std::uint8_t _byte_width;
    const Int _mask;
    const std::uint8_t *_data;
  };

public:
  using value_type = Int;
  using size_type = std::size_t;
  using reference = value_type &;
  using const_reference = const value_type &;
  using iterator = CompactStaticArrayIterator;
  using const_iterator = const CompactStaticArrayIterator;

  /*!
   * Constructs a new CompactStaticArray.
   */
  CompactStaticArray() : _byte_width(0), _size(0), _unrestricted_size(0) {
    RECORD_DATA_STRUCT(0, _struct);
  }

  /*!
   * Constructs a new CompactStaticArray.
   *
   * @param byte_width The number of bytes needed to store the largest integer in the array.
   * @param size The number of values to store.
   */
  CompactStaticArray(const std::uint8_t byte_width, const std::size_t size) {
    KASSERT(byte_width <= 8);
    RECORD_DATA_STRUCT(0, _struct);

    resize(byte_width, size);
  }

  /*!
   * Constructs a new CompactStaticArray.
   *
   * @param byte_width The number of bytes needed to store the largest integer in the array.
   * @param actual_size The number of bytes that the compact representation in memory uses.
   * @param data The pointer to the memory location where the data is compactly stored.
   */
  CompactStaticArray(
      const std::uint8_t byte_width,
      const std::size_t actual_size,
      std::unique_ptr<std::uint8_t[]> data
  )
      : _byte_width(byte_width),
        _size(actual_size),
        _values(std::move(data)),
        _mask(
            (byte_width == 8) ? std::numeric_limits<Int>::max()
                              : (static_cast<std::uint64_t>(1) << (byte_width * 8)) - 1
        ) {
    KASSERT(byte_width <= 8);
    RECORD_DATA_STRUCT(0, _struct);
  }

  CompactStaticArray(const CompactStaticArray &) = delete;
  CompactStaticArray &operator=(const CompactStaticArray &) = delete;

  CompactStaticArray(CompactStaticArray &&) noexcept = default;
  CompactStaticArray &operator=(CompactStaticArray &&) noexcept = default;

  /*!
   * Resizes the array.
   *
   * @param byte_width The number of bytes needed to store the largest integer in the array.
   * @param size The number of values to store.
   */
  void resize(const std::uint8_t byte_width, const std::size_t size) {
    IF_HEAP_PROFILING(
        _struct->size = std::max(_struct->size, byte_width * size + sizeof(Int) - byte_width)
    );

    _byte_width = byte_width;
    _size = byte_width * size + sizeof(Int) - byte_width;
    _unrestricted_size = _size;
    _values = std::make_unique<std::uint8_t[]>(_size);
    _mask = (byte_width == 8) ? std::numeric_limits<Int>::max()
                              : (static_cast<std::uint64_t>(1) << (byte_width * 8)) - 1;
  }

  /*!
   * Restricts the array to a specific size. This operation can be undone by calling the unrestrict
   * method.
   *
   * @param new_size The number of values to be visible.
   */
  void restrict(const std::size_t new_size) {
    _unrestricted_size = _size;
    _size = _byte_width * new_size + sizeof(Int) - _byte_width;
  }

  /*!
   * Undos the previous restriction. It does nothing when the restrict method has previously not
   * been invoked.
   */
  void unrestrict() {
    _size = _unrestricted_size;
  }

  /*!
   * Stores an integer in the array.
   *
   * @param pos The position in the array at which to store the integer.
   * @param value The value to store.
   */
  void write(const std::size_t pos, Int value) {
    std::uint8_t *data = _values.get() + pos * _byte_width;

    for (std::uint8_t i = 0; i < _byte_width; ++i) {
      *data++ = value & 0b11111111;
      value >>= 8;
    }
  }

  /*!
   * Accesses an integer in the array.
   *
   * @param pos The position of the integer in the array to return.
   * @return The integer stored at the position in the array.
   */
  [[nodiscard]] Int operator[](const std::size_t pos) const {
    return *reinterpret_cast<const Int *>(_values.get() + pos * _byte_width) & _mask;
  }

  /*!
   * Returns an interator to the beginning.
   *
   * @return An interator to the beginning.
   */
  [[nodiscard]] CompactStaticArrayIterator begin() const {
    return CompactStaticArrayIterator(_byte_width, _mask, _values.get());
  }

  /*!
   * Returns an interator to the end.
   *
   * @return An interator to the end.
   */
  [[nodiscard]] CompactStaticArrayIterator end() const {
    return CompactStaticArrayIterator(
        _byte_width, _mask, _values.get() + _size - (sizeof(Int) - _byte_width)
    );
  }

  /*!
   * Returns whether the array is empty.
   *
   * @return Whether the array is empty.
   */
  [[nodiscard]] bool empty() const {
    return _size == 0;
  }

  /*!
   * Returns the amount of integers in the array.
   *
   * @return The amount of integers in the array.
   */
  [[nodiscard]] std::size_t size() const {
    return (_size - (sizeof(Int) - _byte_width)) / _byte_width;
  }

  /*!
   * Returns the number of bytes needed to store the largest integer in the array.
   *
   * @return The number of bytes needed to store the largest integer in the array.
   */
  [[nodiscard]] std::uint8_t byte_width() const {
    return _byte_width;
  }

  /*!
   * Returns the amount of bytes the compact array allocated.
   *
   * @return The amount of bytes the compact array allocated.
   */
  [[nodiscard]] std::size_t allocated_size() const {
    return _size;
  }

  /*!
   * Returns a pointer to the memory location where the data is compactly stored.
   *
   * @returns A pointer to the memory location where the data is compactly stored.
   */
  [[nodiscard]] const std::uint8_t *data() const {
    return _values.get();
  }

private:
  std::uint8_t _byte_width;
  std::size_t _size;
  std::size_t _unrestricted_size;
  std::unique_ptr<std::uint8_t[]> _values;
  Int _mask;

  IF_HEAP_PROFILING(heap_profiler::DataStructure *_struct);
};

}; // namespace kaminpar
