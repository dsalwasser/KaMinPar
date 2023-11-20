/*******************************************************************************
 * Compressed static graph representations.
 *
 * @file:   compressed_graph.h
 * @author: Daniel Salwasser
 * @date:   07.11.2023
 ******************************************************************************/
#pragma once

#include <deque>
#include <iterator>
#include <vector>

#include <kassert/kassert.hpp>

#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/ranges.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

/*!
 * A compressed static graph that stores the nodes and edges in a compressed adjacency array. It
 * uses variable length encoding, gap encoding and interval encoding to compress the edge array.
 *
 * @tparam VarLengthCodec The namespace that contains functions to encode and decode variable length
 * integers.
 * @tparam IntervalEncoding Whether interval encoding should be used as a compression method.
 */
template <typename VarLengthCodec, bool IntervalEncoding = true> class CompressedGraph {
public:
  using NodeID = ::kaminpar::shm::NodeID;
  using NodeWeight = ::kaminpar::shm::NodeWeight;
  using EdgeID = ::kaminpar::shm::EdgeID;
  using EdgeWeight = ::kaminpar::shm::EdgeWeight;

  class CompressedEdgesRange {
  public:
    class iterator {
    public:
      using iterator_category = std::input_iterator_tag;
      using value_type = NodeID;
      using difference_type = std::make_signed_t<NodeID>;
      using pointer = value_type *;
      using reference = value_type &;

      iterator(const NodeID node, const bool uses_intervals, const std::uint8_t *ptr)
          : _node(node),
            _uses_intervals(uses_intervals),
            _ptr(ptr) {
        if constexpr (IntervalEncoding) {
          if (_uses_intervals) {
            auto [interval_count, interval_count_len] =
                VarLengthCodec::template decode<NodeID>(_ptr);
            _ptr += interval_count_len;

            _interval_count = interval_count;
          }
        }
      }

      value_type operator*() {
        if constexpr (IntervalEncoding) {
          if (_uses_intervals) {
            if (_cur_interval_index < _cur_interval_len) {
              return _cur_left_extreme + _cur_interval_index;
            }

            if (_cur_interval < _interval_count) {
              auto [left_extreme_gap, left_extreme_gap_len] =
                  VarLengthCodec::template decode<NodeID>(_ptr);
              auto [interval_length_gap, interval_length_gap_len] =
                  VarLengthCodec::template decode<NodeID>(_ptr + left_extreme_gap_len);

              _len = left_extreme_gap_len + interval_length_gap_len;

              _cur_left_extreme = left_extreme_gap + _previous_right_extreme - 2;
              _cur_interval_len = interval_length_gap + kIntervalLengthTreshold;
              _previous_right_extreme = _cur_left_extreme + _cur_interval_len - 1;

              _cur_interval_index = _cur_interval_len;
              return _cur_left_extreme;
            }
          }
        }

        if (_first) {
          _first = false;

          auto [first_gap, first_gap_len] = VarLengthCodec::template decode_signed<NodeID>(_ptr);
          const NodeID first_adjacent_node = first_gap + _node;

          _len = first_gap_len;
          _prev_adjacent_node = first_adjacent_node;
          return first_adjacent_node;
        }

        auto [gap, gap_len] = VarLengthCodec::template decode<NodeID>(_ptr);
        const NodeID adjacent_node = gap + _prev_adjacent_node;

        _len = gap_len;
        _prev_adjacent_node = adjacent_node;
        return adjacent_node;
      }

      iterator &operator++() {
        if constexpr (IntervalEncoding) {
          if (_uses_intervals) {
            if (_cur_interval_index < _cur_interval_len) {
              _cur_interval_index += 1;
              return *this;
            }

            if (_cur_interval < _interval_count) {
              _cur_interval += 1;
              _cur_interval_index = 1;
              _ptr += _len;
              return *this;
            }
          }
        }

        _ptr += _len;
        return *this;
      }

      iterator operator++(int) {
        auto tmp = *this;
        ++*this;
        return tmp;
      }

      bool operator==(const iterator &other) const {
        if constexpr (IntervalEncoding) {
          return _ptr == other._ptr && _cur_interval == _interval_count &&
                 _cur_interval_index == _cur_interval_len;
        }

        return _ptr == other._ptr;
      }

      bool operator!=(const iterator &other) const {
        return !(*this == other);
      }

    private:
      const NodeID _node;
      const bool _uses_intervals;
      const std::uint8_t *_ptr;

      // Interval encoding
      NodeID _interval_count = 0;
      NodeID _cur_interval = 0;
      NodeID _cur_left_extreme = 0;
      NodeID _cur_interval_len = 0;
      NodeID _cur_interval_index = 0;
      NodeID _previous_right_extreme = 2;

      // Gap encoding
      bool _first = true;
      std::size_t _len = 0;
      NodeID _prev_adjacent_node;
    };

    CompressedEdgesRange(
        const NodeID node,
        const bool uses_intervals,
        const std::uint8_t *begin,
        const std::uint8_t *end
    )
        : _begin(node, uses_intervals, begin),
          _end(node, false, end) {}

    iterator begin() const {
      return _begin;
    }
    iterator end() const {
      return _end;
    }

  private:
    iterator _begin;
    iterator _end;
  };

  /*!
   * The minimum length of an interval to encode if interval encoding is used.
   */
  static constexpr std::size_t kIntervalLengthTreshold = 3;

  /**
   * Compresses a graph.
   *
   * @param graph The graph to compress.
   * @return The compressed input graph.
   */
  static CompressedGraph<VarLengthCodec, IntervalEncoding> compress(const Graph &graph) {
    SCOPED_HEAP_PROFILER("Compress graph");
    SCOPED_TIMER("Compress graph");

    auto iterate = [&](auto &&handle_node,
                       auto &&handle_interval,
                       auto &&handle_first_gap,
                       auto &&handle_remaining_gap) {
      std::vector<NodeID> buffer;

      for (const NodeID node : graph.nodes()) {
        handle_node(node);

        const NodeID degree = graph.degree(node);
        if (degree == 0) {
          continue;
        }

        for (const NodeID adjacent_node : graph.adjacent_nodes(node)) {
          buffer.push_back(adjacent_node);
        }

        // Sort the adjacent nodes in ascending order.
        std::sort(buffer.begin(), buffer.end());

        // Find intervals [i, j] of consecutive adjacent nodes i, i + 1, ..., j - 1, j of length at
        // least kIntervalLengthTreshold. Instead of storing all nodes, only store a representation
        // of the left extreme i and the length j - i + 1. Left extremes are compressed using the
        // differences between each left extreme and the previous right extreme minus 2 (because
        // there must be at least one integer between the end of an interval and the beginning of
        // the next one), except the first left extreme which is stored directly. The lengths are
        // decremented by kIntervalLengthTreshold, the minimum length of an interval.
        if constexpr (IntervalEncoding) {
          if (buffer.size() > 1) {
            NodeID previous_right_extreme = 2;
            std::size_t interval_len = 1;

            NodeID prev_adjacent_node = *buffer.begin();
            for (auto iter = buffer.begin() + 1; iter != buffer.end(); ++iter) {
              const NodeID adjacent_node = *iter;

              if (prev_adjacent_node + 1 == adjacent_node) {
                interval_len++;

                // The interval ends if there are no more nodes or the next node is not the
                // increment of the current node.
                if (iter + 1 == buffer.end() || adjacent_node + 1 != *(iter + 1)) {
                  if (interval_len >= kIntervalLengthTreshold) {
                    const NodeID left_extreme = adjacent_node + 1 - interval_len;
                    const NodeID left_extreme_gap = left_extreme + 2 - previous_right_extreme;
                    const std::size_t interval_length_gap = interval_len - kIntervalLengthTreshold;

                    handle_interval(left_extreme_gap, interval_length_gap);

                    previous_right_extreme = adjacent_node;
                    iter = buffer.erase(iter - interval_len + 1, iter + 1);
                    if (iter == buffer.end()) {
                      break;
                    }
                  }

                  interval_len = 1;
                }
              }

              prev_adjacent_node = adjacent_node;
            }

            // If all incident edges have been compressed using intervals then gap encoding cannot
            // be applied. Thus, go to the next node.
            if (buffer.empty()) {
              continue;
            }
          }
        }

        // Store the remaining adjacent node using gap encoding. That is instead of storing the
        // nodes v_1, v_2, ..., v_{k - 1}, v_k directly, store the gaps v_1 - u, v_2 - v_1, ..., v_k
        // - v_{k - 1} between the nodes, where u is the source node. Note that all gaps except the
        // first one have to be positive as we sorted the nodes in ascending order. Thus, only for
        // the first gap the sign is additionally stored.
        const NodeID first_adjacent_node = *buffer.begin();
        // TODO: Does the value range cover everything s.t. a over- or underflow cannot happen?
        const std::make_signed_t<NodeID> first_gap = first_adjacent_node - node;
        handle_first_gap(first_gap);

        NodeID prev_adjacent_node = first_adjacent_node;
        const auto iter_end = buffer.end();
        for (auto iter = buffer.begin() + 1; iter != iter_end; ++iter) {
          const NodeID adjacent_node = *iter;
          const NodeID gap = adjacent_node - prev_adjacent_node;

          handle_remaining_gap(gap);
          prev_adjacent_node = adjacent_node;
        }

        buffer.clear();
      }
    };

    // First iterate over all nodes and their adjacent nodes. In the process calculate the number of
    // intervalls to store compressed for each node and store the number temporarily in the nodes
    // array. Additionally calculate the needed capacity for the compressed edge array.
    RECORD("nodes") StaticArray<EdgeID> nodes(graph.n() + 1);

    NodeID cur_node;
    std::size_t edge_capacity = 0;
    iterate(
        [&](auto node) {
          cur_node = node;

          if constexpr (IntervalEncoding) {
            edge_capacity += VarLengthCodec::length_marker(graph.degree(node));
          } else {
            edge_capacity += VarLengthCodec::length(graph.degree(node));
          }
        },
        [&](auto left_extreme_gap, auto interval_length_gap) {
          nodes[cur_node] += 1;
          edge_capacity += VarLengthCodec::length(left_extreme_gap);
          edge_capacity += VarLengthCodec::length(interval_length_gap);
        },
        [&](auto first_gap) { edge_capacity += VarLengthCodec::length_signed(first_gap); },
        [&](auto gap) { edge_capacity += VarLengthCodec::length(gap); }
    );

    if constexpr (IntervalEncoding) {
      auto iter_end = nodes.end();
      for (auto iter = nodes.begin(); iter + 1 != iter_end; ++iter) {
        const EdgeID number_of_intervalls = *iter;

        if (number_of_intervalls > 0) {
          edge_capacity += VarLengthCodec::length(number_of_intervalls);
        }
      }
    }

    // In the second iteration fill the nodes and compressed edge array with data.
    RECORD("compressed_edges") StaticArray<std::uint8_t> compressed_edges(edge_capacity);
    std::size_t interval_count = 0;

    uint8_t *edges = compressed_edges.data();
    iterate(
        [&](auto node) {
          const EdgeID number_of_intervalls = nodes[node];

          nodes[node] = static_cast<EdgeID>(edges - compressed_edges.data());
          if constexpr (IntervalEncoding) {
            edges += VarLengthCodec::encode_with_marker(
                graph.degree(node), number_of_intervalls > 0, edges
            );
          } else {
            edges += VarLengthCodec::encode(graph.degree(node), edges);
          }

          if constexpr (IntervalEncoding) {
            if (number_of_intervalls > 0) {
              edges += VarLengthCodec::encode(number_of_intervalls, edges);
              interval_count++;
            }
          }
        },
        [&](auto left_extreme_gap, auto interval_length_gap) {
          edges += VarLengthCodec::encode(left_extreme_gap, edges);
          edges += VarLengthCodec::encode(interval_length_gap, edges);
        },
        [&](auto first_gap) { edges += VarLengthCodec::encode_signed(first_gap, edges); },
        [&](auto gap) { edges += VarLengthCodec::encode(gap, edges); }
    );
    nodes[nodes.size() - 1] = compressed_edges.size();

    return CompressedGraph<VarLengthCodec, IntervalEncoding>(
        std::move(nodes), std::move(compressed_edges), graph.m(), interval_count
    );
  }

  /*!
   * Constructs a new compressed graph.
   *
   * @param nodes The node array which stores for each node the offset in the compressed edges array
   * of the first edge.
   * @param compressed_edges The edge array which stores the edges for each node in a compressed
   * format.
   * @param edge_count The number of edges stored in the compressed edge array.
   * @param interval_count The number of nodes which use interval encoding.
   */
  explicit CompressedGraph(
      StaticArray<EdgeID> nodes,
      StaticArray<std::uint8_t> compressed_edges,
      std::size_t edge_count,
      std::size_t interval_count
  )
      : _nodes(std::move(nodes)),
        _compressed_edges(std::move(compressed_edges)),
        _edge_count(edge_count),
        _interval_count(interval_count) {
    KASSERT(IntervalEncoding || interval_count == 0);
  };

  /*!
   * Returns the number of nodes of the graph.
   *
   * @return The number of nodes of the graph.
   */
  [[nodiscard]] NodeID n() const {
    return static_cast<NodeID>(_nodes.size() - 1);
  };

  /*!
   * Returns the number of edges of the graph.
   *
   * @return The number of edges of the graph.
   */
  [[nodiscard]] EdgeID m() const {
    return static_cast<EdgeID>(_edge_count);
  }

  /*!
   * Returns a range that contains all nodes of the graph.
   *
   * @return A range that contains all nodes of the graph.
   */
  [[nodiscard]] IotaRange<NodeID> nodes() const {
    return IotaRange(static_cast<NodeID>(0), n());
  }

  /**
   * Returns the degree of a node.
   *
   * @param node The node for which the degree is to be returned.
   * @return The degree of the node.
   */
  [[nodiscard]] NodeID degree(const NodeID node) const {
    const std::uint8_t *data = _compressed_edges.data() + _nodes[node];
    auto [degree, _] = VarLengthCodec::template decode<NodeID>(data);
    return degree;
  }

  [[nodiscard]] CompressedEdgesRange adjacent_nodes(const NodeID node) const {
    const std::uint8_t *begin = _compressed_edges.data() + _nodes[node];
    const std::uint8_t *end = _compressed_edges.data() + _nodes[node + 1];

    if constexpr (IntervalEncoding) {
      auto [degree, uses_intervals, degree_len] =
          VarLengthCodec::template decode_with_marker<NodeID>(begin);
      begin += degree_len;

      return CompressedEdgesRange(node, uses_intervals, begin, end);
    }

    auto [degree, degree_len] = VarLengthCodec::template decode<NodeID>(begin);
    begin += degree_len;

    return CompressedEdgesRange(node, false, begin, end);
  }

  /**
   * Returns the number of nodes which use interval encoding.
   *
   * @returns The number of nodes which use interval encoding.
   */
  [[nodiscard]] std::size_t interval_count() const {
    return _interval_count;
  }

  /*!
   * Returns the amount memory in bytes used by the data structure.
   *
   * @return The amount memory in bytes used by the data structure.
   */
  [[nodiscard]] std::size_t used_memory() const {
    return _nodes.size() * sizeof(EdgeID) + _compressed_edges.size() * sizeof(std::uint8_t);
  }

  /**
   * Returns the array of raw nodes.
   *
   * @return The array of raw nodes.
   */
  [[nodiscard]] const StaticArray<EdgeID> &raw_nodes() const {
    return _nodes;
  }

  /**
   * Returns the array of raw compressed edges.
   *
   * @return The array of raw compressed edges.
   */
  [[nodiscard]] const StaticArray<std::uint8_t> &raw_compressed_edges() const {
    return _compressed_edges;
  }

private:
  StaticArray<EdgeID> _nodes;
  StaticArray<std::uint8_t> _compressed_edges;
  const std::size_t _edge_count;
  const std::size_t _interval_count;
};

} // namespace kaminpar::shm
