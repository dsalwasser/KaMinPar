#pragma once

#ifdef KAMINPAR_ENABLE_PAPI

#include <vector>

namespace kaminpar::papi {
using EventSetID = int;
using EventID = int;
using Counter = long long;

enum class EventKind {
  INTEGER_INSTRUCTIONS,
  FLOAT_INSTRUCTIONS,
  LOAD_INSTRUCTIONS,
  STORE_INSTRUCTIONS,
  BRANCH_INSTRUCTIONS,
  CONDITIONAL_BRANCH_INSTRUCTIONS,
  TOTAL_INSTRUCTIONS,

  CONDITIONAL_BRANCH_INSTRUCTIONS_TAKEN,
  CONDITIONAL_BRANCH_INSTRUCTIONS_NOT_TAKEN,
  CONDITIONAL_BRANCH_INSTRUCTIONS_MISPREDICTED,
  CONDITIONAL_BRANCH_INSTRUCTIONS_PREDICTED,

  L1_DATA_CACHE_MISS,
  L1_DATA_CACHE_HIT,
  L1_DATA_CACHE_ACCESS,

  L1_INSTRUCTION_CACHE_MISS,
  L1_INSTRUCTION_CACHE_HIT,
  L1_INSTRUCTION_CACHE_ACCESS,

  L1_TOTAL_CACHE_MISS,
  L1_TOTAL_CACHE_HIT,
  L1_TOTAL_CACHE_ACCESS,

  L2_DATA_CACHE_MISS,
  L2_DATA_CACHE_HIT,
  L2_DATA_CACHE_ACCESS,

  L2_INSTRUCTION_CACHE_MISS,
  L2_INSTRUCTION_CACHE_HIT,
  L2_INSTRUCTION_CACHE_ACCESS,

  L2_TOTAL_CACHE_MISS,
  L2_TOTAL_CACHE_HIT,
  L2_TOTAL_CACHE_ACCESS,

  L3_DATA_CACHE_MISS,
  L3_DATA_CACHE_HIT,
  L3_DATA_CACHE_ACCESS,

  L3_INSTRUCTION_CACHE_MISS,
  L3_INSTRUCTION_CACHE_HIT,
  L3_INSTRUCTION_CACHE_ACCESS,

  L3_TOTAL_CACHE_MISS,
  L3_TOTAL_CACHE_HIT,
  L3_TOTAL_CACHE_ACCESS,
};

class EventSet {
public:
  EventSet(EventSetID id, std::vector<EventKind> event_kinds);
  ~EventSet();

  EventSet(const EventSet &) = delete;
  EventSet &operator=(const EventSet &) = delete;

  EventSet(EventSet &&) noexcept = default;
  EventSet &operator=(EventSet &&) noexcept = default;

  void start();
  void read();
  void reset();
  void stop();

  [[nodiscard]] std::size_t get_counter(EventKind kind) const;

private:
  EventSetID id;
  std::vector<EventKind> event_kinds;
  std::vector<Counter> counters;
};

void initialize();
void initialize_thread();
EventSet create_event_set(std::vector<EventKind> event_kinds);

} // namespace kaminpar::papi

#define PAPI_INIT() kaminpar::papi::initialize()
#define PAPI_INIT_THREAD() kaminpar::papi::initialize_thread()

#else

#define PAPI_INIT()
#define PAPI_INIT_THREAD()

#endif
