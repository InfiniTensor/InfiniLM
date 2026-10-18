#pragma once

#include "op_type.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace infinilm::execution::trace {

// ============================================================
// OpTraceEntry — a single recorded op invocation
// ============================================================

/// Compact record of one operator invocation.
struct OpTraceEntry {
    OpType op_type = OpType::UNKNOWN;

    // Tensor shape summary (up to 4 dims for the primary input tensor)
    static constexpr size_t MAX_DIMS = 4;
    uint32_t ndim = 0;
    uint32_t shape[MAX_DIMS] = {};

    // Data type of the primary input.
    uint8_t dtype = 0;

    // Device info
    uint8_t device_type = 0;
    int8_t device_id = -1;

    // Steady-clock timestamp.
    uint64_t timestamp_ns = 0;

    /// Fill shape from a shape vector.
    void setShape(const size_t *dims, size_t n) {
        ndim = static_cast<uint32_t>(n > MAX_DIMS ? MAX_DIMS : n);
        for (uint32_t i = 0; i < ndim; ++i) {
            shape[i] = static_cast<uint32_t>(dims[i]);
        }
    }

    /// Get current steady_clock timestamp in nanoseconds.
    static uint64_t now() {
        return static_cast<uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count());
    }
};

// ============================================================
// OpTraceRing — ring buffer for op trace entries
// ============================================================

/// A fixed-capacity ring buffer. Callers must serialize writes.
class OpTraceRing {
public:
    static constexpr size_t DEFAULT_CAPACITY = 256;

    explicit OpTraceRing(size_t capacity = DEFAULT_CAPACITY)
        : capacity_(capacity),
          entries_(capacity),
          write_pos_(0) {
    }

    /// Record a new op trace entry.
    void write(const OpTraceEntry &entry) {
        size_t pos = write_pos_.load(std::memory_order_relaxed);
        entries_[pos % capacity_] = entry;
        write_pos_.store(pos + 1, std::memory_order_release);
    }

private:
    size_t capacity_;
    std::vector<OpTraceEntry> entries_;
    std::atomic<size_t> write_pos_;
};

// ============================================================
// Global OpTrace singleton access
// ============================================================

/// Get the global OpTraceRing instance.
/// This is the primary entry point for recording op traces.
OpTraceRing &getGlobalOpTrace();

/// Record an op invocation to the global trace ring.
/// Called by `INFINILM_TRACE_OP` when `ENABLE_MUTUAL_AWARENESS` is defined.
inline void traceOp(OpType op_type,
                    const size_t *shape, size_t ndim,
                    uint8_t dtype,
                    uint8_t device_type, int8_t device_id) {
    OpTraceEntry entry;
    entry.op_type = op_type;
    entry.setShape(shape, ndim);
    entry.dtype = dtype;
    entry.device_type = device_type;
    entry.device_id = device_id;
    entry.timestamp_ns = OpTraceEntry::now();
    getGlobalOpTrace().write(entry);
}

} // namespace infinilm::execution::trace
