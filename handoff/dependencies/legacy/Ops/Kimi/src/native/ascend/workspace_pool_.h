#ifndef INFINI_OPS_ASCEND_WORKSPACE_POOL__H_
#define INFINI_OPS_ASCEND_WORKSPACE_POOL__H_

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "acl/acl.h"

namespace infini::ops::ascend {

struct WorkspaceArena {
  void* buf = nullptr;

  uint64_t capacity = 0;
};

class WorkspacePool {
 public:
  // Ensure the arena for `(stream, slot)` has at least `needed` bytes.
  //
  // The `slot` parameter defaults to `"workspace"` for backward
  // compatibility. Operators needing a separate temp arena pass
  // `"temp"`.
  WorkspaceArena& Ensure(aclrtStream stream, uint64_t needed,
                         const char* slot = "workspace") {
    // Thread-local fast path: a small flat array of recently used
    // `(stream, slot, arena*)` triples. In practice operators use at
    // most 2-3 slots, so linear scan is sufficient — no heap
    // allocation on the hot path.
    struct CacheEntry {
      aclrtStream stream = nullptr;
      const char* slot = nullptr;
      WorkspaceArena* arena = nullptr;
    };
    static constexpr int kCacheSize = 4;
    thread_local CacheEntry cache[kCacheSize] = {};

    for (int i = 0; i < kCacheSize; ++i) {
      auto& e = cache[i];

      if (e.stream == stream && e.slot != nullptr &&
          std::strcmp(e.slot, slot) == 0 && e.arena != nullptr &&
          needed <= e.arena->capacity) {
        return *e.arena;
      }
    }

    // Slow path: look up arena in the map under lock.
    assert(!capturing_ &&
           "`WorkspacePool`: `aclrtMalloc` on slow path during graph "
           "capture. Ensure all operators run at least once during "
           "eager warmup.");

    std::lock_guard<std::mutex> lock(mutex_);

    SlotKey key{stream, slot};
    auto& owned = arenas_[key];

    if (!owned) {
      owned = std::make_unique<WorkspaceArena>();
    }

    auto* arena = owned.get();

    if (needed > arena->capacity) {
      const auto new_capacity = NextCapacity(arena->capacity, needed);
      void* new_buf = nullptr;

      if (new_capacity > 0) {
        auto ret =
            aclrtMalloc(&new_buf, new_capacity, ACL_MEM_MALLOC_NORMAL_ONLY);
        assert(ret == ACL_SUCCESS && "`WorkspacePool`: `aclrtMalloc` failed");
      }

      // CANN RI graphs capture the raw workspace address. A graph may outlive
      // a later arena growth, so freeing the old allocation here would leave
      // the captured graph with a dangling pointer. Keep old generations alive
      // until the process-level pool is destroyed.
      if (arena->buf != nullptr) {
        retained_arenas_.push_back(*arena);
      }

      arena->buf = new_buf;
      arena->capacity = new_capacity;
    }

    // Insert into the thread-local cache (evict oldest).
    for (int i = kCacheSize - 1; i > 0; --i) {
      cache[i] = cache[i - 1];
    }
    cache[0] = {stream, slot, arena};

    return *arena;
  }

  // Set to true before NPUGraph capture, false after.  When true,
  // the slow path (which calls `aclrtMalloc`) triggers an assert
  // failure — a safety net against accidental device allocations
  // being recorded into the graph.
  void set_capture_mode(bool capturing) { capturing_ = capturing; }

  ~WorkspacePool() {
    for (auto& arena : retained_arenas_) {
      int32_t dev_id = -1;
      if (aclrtGetDevice(&dev_id) == ACL_SUCCESS) {
        aclrtFree(arena.buf);
      } else {
        fprintf(stderr,
                "[InfiniOps] `WorkspacePool`: CANN runtime already "
                "finalized, skipping retained `aclrtFree` (%" PRIu64
                " bytes leaked).\n",
                arena.capacity);
      }
    }
    for (auto& [key, arena] : arenas_) {
      if (arena && arena->capacity > 0) {
        // The CANN runtime may already be torn down when this static
        // destructor runs.  `aclrtGetDevice` fails in that case —
        // skip the free to avoid glibc "double free" abort.
        int32_t dev_id = -1;

        if (aclrtGetDevice(&dev_id) == ACL_SUCCESS) {
          aclrtFree(arena->buf);
        } else {
          fprintf(stderr,
                  "[InfiniOps] `WorkspacePool`: CANN runtime already "
                  "finalized, skipping `aclrtFree` (%" PRIu64
                  " bytes leaked).\n",
                  arena->capacity);
        }
      }
    }
  }

 private:
  static uint64_t NextCapacity(uint64_t current, uint64_t needed) {
    if (current == 0) {
      return needed;
    }

    const auto growth = current / 2;
    if (current > std::numeric_limits<uint64_t>::max() - growth) {
      return needed;
    }

    return std::max(needed, current + growth);
  }

  struct SlotKey {
    aclrtStream stream;
    std::string slot;

    bool operator==(const SlotKey& o) const {
      return stream == o.stream && slot == o.slot;
    }
  };

  struct SlotKeyHash {
    size_t operator()(const SlotKey& k) const {
      auto h1 = std::hash<void*>{}(static_cast<void*>(k.stream));
      auto h2 = std::hash<std::string>{}(k.slot);

      return h1 ^ (h2 << 1);
    }
  };

  std::unordered_map<SlotKey, std::unique_ptr<WorkspaceArena>, SlotKeyHash>
      arenas_;

  std::vector<WorkspaceArena> retained_arenas_;

  std::mutex mutex_;

  bool capturing_ = false;
};

inline WorkspacePool& GetWorkspacePool() {
  static WorkspacePool pool;

  return pool;
}

}  // namespace infini::ops::ascend

#endif
