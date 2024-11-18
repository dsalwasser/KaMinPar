#include "kaminpar-common/papi.h"

#ifdef KAMINPAR_ENABLE_PAPI

#include <stdexcept>
#include <string>

#include <papi.h>
#include <pthread.h>
#include <tbb/task_arena.h>

#include "kaminpar-common/logger.h"

#define COMPILER_BARRIER() asm volatile("" ::: "memory")

namespace kaminpar::papi {

namespace {
SET_DEBUG(false);

EventID convert_to_papi_event(const EventKind event) {
  switch (event) {
  case EventKind::INTEGER_INSTRUCTIONS:
    return PAPI_INT_INS;
  case EventKind::FLOAT_INSTRUCTIONS:
    return PAPI_FP_INS;
  case EventKind::LOAD_INSTRUCTIONS:
    return PAPI_LD_INS;
  case EventKind::STORE_INSTRUCTIONS:
    return PAPI_SR_INS;
  case EventKind::BRANCH_INSTRUCTIONS:
    return PAPI_BR_INS;
  case EventKind::CONDITIONAL_BRANCH_INSTRUCTIONS:
    return PAPI_BR_CN;
  case EventKind::TOTAL_INSTRUCTIONS:
    return PAPI_TOT_INS;

  case EventKind::CONDITIONAL_BRANCH_INSTRUCTIONS_TAKEN:
    return PAPI_BR_TKN;
  case EventKind::CONDITIONAL_BRANCH_INSTRUCTIONS_NOT_TAKEN:
    return PAPI_BR_NTK;
  case EventKind::CONDITIONAL_BRANCH_INSTRUCTIONS_MISPREDICTED:
    return PAPI_BR_MSP;
  case EventKind::CONDITIONAL_BRANCH_INSTRUCTIONS_PREDICTED:
    return PAPI_BR_PRC;

  case EventKind::L1_DATA_CACHE_MISS:
    return PAPI_L1_DCM;
  case EventKind::L1_DATA_CACHE_HIT:
    return PAPI_L1_DCH;
  case EventKind::L1_DATA_CACHE_ACCESS:
    return PAPI_L1_DCA;

  case EventKind::L1_INSTRUCTION_CACHE_MISS:
    return PAPI_L1_ICM;
  case EventKind::L1_INSTRUCTION_CACHE_HIT:
    return PAPI_L1_ICH;
  case EventKind::L1_INSTRUCTION_CACHE_ACCESS:
    return PAPI_L1_ICA;

  case EventKind::L1_TOTAL_CACHE_MISS:
    return PAPI_L1_TCM;
  case EventKind::L1_TOTAL_CACHE_HIT:
    return PAPI_L1_TCH;
  case EventKind::L1_TOTAL_CACHE_ACCESS:
    return PAPI_L1_TCA;

  case EventKind::L2_DATA_CACHE_MISS:
    return PAPI_L2_DCM;
  case EventKind::L2_DATA_CACHE_HIT:
    return PAPI_L2_DCH;
  case EventKind::L2_DATA_CACHE_ACCESS:
    return PAPI_L2_DCA;

  case EventKind::L2_INSTRUCTION_CACHE_MISS:
    return PAPI_L2_ICM;
  case EventKind::L2_INSTRUCTION_CACHE_HIT:
    return PAPI_L2_ICH;
  case EventKind::L2_INSTRUCTION_CACHE_ACCESS:
    return PAPI_L2_ICA;

  case EventKind::L2_TOTAL_CACHE_MISS:
    return PAPI_L2_TCM;
  case EventKind::L2_TOTAL_CACHE_HIT:
    return PAPI_L2_TCH;
  case EventKind::L2_TOTAL_CACHE_ACCESS:
    return PAPI_L2_TCA;

  case EventKind::L3_DATA_CACHE_MISS:
    return PAPI_L3_DCM;
  case EventKind::L3_DATA_CACHE_HIT:
    return PAPI_L3_DCH;
  case EventKind::L3_DATA_CACHE_ACCESS:
    return PAPI_L3_DCA;

  case EventKind::L3_INSTRUCTION_CACHE_MISS:
    return PAPI_L3_ICM;
  case EventKind::L3_INSTRUCTION_CACHE_HIT:
    return PAPI_L3_ICH;
  case EventKind::L3_INSTRUCTION_CACHE_ACCESS:
    return PAPI_L3_ICA;

  case EventKind::L3_TOTAL_CACHE_MISS:
    return PAPI_L3_TCM;
  case EventKind::L3_TOTAL_CACHE_HIT:
    return PAPI_L3_TCH;
  case EventKind::L3_TOTAL_CACHE_ACCESS:
    return PAPI_L3_TCA;

  default:
    throw std::invalid_argument{"Failed to map event to PAPI code"};
  };
}

} // namespace

EventSet::EventSet(EventSetID id, std::vector<EventKind> event_kinds)
    : id(id),
      event_kinds(std::move(event_kinds)),
      counters(this->event_kinds.size(), 0) {}

EventSet::~EventSet() {
  if (PAPI_cleanup_eventset(id) != PAPI_OK) {
    LOG_ERROR << "Failed to clean up event set";
    return;
  }

  if (PAPI_destroy_eventset(&id) != PAPI_OK) {
    LOG_ERROR << "Failed to destroy event set";
    return;
  }
}

void EventSet::start() {
  COMPILER_BARRIER();
  if (PAPI_start(id) != PAPI_OK) {
    throw std::runtime_error{"Failed to start event set"};
  }
  COMPILER_BARRIER();
}

void EventSet::read() {
  COMPILER_BARRIER();
  if (PAPI_read(id, counters.data()) != PAPI_OK) {
    throw std::runtime_error{"Failed to read event set"};
  }
  COMPILER_BARRIER();
}

void EventSet::reset() {
  COMPILER_BARRIER();
  if (PAPI_reset(id) != PAPI_OK) {
    throw std::runtime_error{"Failed to reset event set"};
  }
  COMPILER_BARRIER();
}

void EventSet::stop() {
  COMPILER_BARRIER();
  if (PAPI_stop(id, counters.data()) != PAPI_OK) {
    throw std::runtime_error{"Failed to stop event set"};
  }
  COMPILER_BARRIER();
}

std::size_t EventSet::get_counter(EventKind kind) const {
  for (std::size_t i = 0; i < event_kinds.size(); ++i) {
    if (event_kinds[i] == kind) {
      return counters[i];
    }
  }

  throw std::runtime_error{"Failed to get counter as the provided event was not registered"};
}

void initialize() {
  if (PAPI_is_initialized()) {
    return;
  }

  if (const auto ret = PAPI_library_init(PAPI_VER_CURRENT); ret != PAPI_VER_CURRENT) {
    throw std::runtime_error{"Failed to initialize PAPI"};
  } else {
    DBG << "Initialized PAPI version: " << PAPI_VERSION_MAJOR(ret) << '.' << PAPI_VERSION_MINOR(ret)
        << '.' << PAPI_VERSION_REVISION(ret) << '.' << PAPI_VERSION_INCREMENT(ret);
  }

  if (PAPI_thread_init(pthread_self) != PAPI_OK) {
    throw std::runtime_error{"Failed to initialize PAPI thread support"};
    return;
  } else {
    DBG << "Initialzied PAPI thread support";
  }
}

void initialize_thread() {
  static thread_local bool registered = false;
  if (registered) {
    return;
  }

  if (PAPI_register_thread() != PAPI_OK) {
    throw std::runtime_error{"Failed to register thread for PAPI"};
  }

  DBG << "Registered thread (tid: " << PAPI_thread_id()
      << ", tbb_tid: " << tbb::this_task_arena::current_thread_index() << ") for PAPI";
  registered = true;
}

EventSet create_event_set(std::vector<EventKind> event_kinds) {
  EventSetID event_set_id = PAPI_NULL;
  if (PAPI_create_eventset(&event_set_id) != PAPI_OK) {
    throw std::runtime_error{"Failed to create PAPI event set"};
  }

  for (const auto event_kind : event_kinds) {
    const EventID event_id = convert_to_papi_event(event_kind);

    const auto convert_event_to_string = [&]() {
      return "(eid:" + std::to_string(event_id) + '/' +
             std::to_string(static_cast<int>(event_kind)) + ')';
    };

    if (PAPI_query_event(event_id) != PAPI_OK) {
      throw std::runtime_error{
          "Failed to add PAPI event " + convert_event_to_string() +
          " as it does not exist on this architecture"
      };
    }

    if (PAPI_add_event(event_set_id, event_id) != PAPI_OK) {
      throw std::runtime_error{"Failed to add PAPI event " + convert_event_to_string()};
    }
  }

  return {event_set_id, std::move(event_kinds)};
}

} // namespace kaminpar::papi

#endif
