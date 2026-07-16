#pragma once

#include "op_type.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace infinicore::analyzer {

// ============================================================
// OpTraceEntry — a single recorded op invocation
// ============================================================

/// Compact record of one operator invocation for phase detection.
/// Designed to be small (~80 bytes) and cheap to fill.
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

    // Timestamp (nanoseconds since epoch, from steady_clock)
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
// OpTraceRing — lock-free ring buffer for op trace entries
// ============================================================

/// A fixed-capacity ring buffer for OpTraceEntry.
/// Single-producer (op execution thread) friendly.
/// Reader can safely read a snapshot via getRecentEntries().
///
/// Thread safety:
/// - write() is safe to call from the single producer thread
///   (typical in InfiniCore where ops are dispatched on one thread).
/// - getRecentEntries() takes a snapshot and is safe to call from
///   any thread (may see a partially written entry at the boundary,
///   which is acceptable for heuristic phase detection).
class OpTraceRing {
public:
    static constexpr size_t DEFAULT_CAPACITY = 256;

    explicit OpTraceRing(size_t capacity = DEFAULT_CAPACITY)
        : capacity_(capacity),
          entries_(capacity),
          write_pos_(0),
          total_count_(0) {
    }

    /// Record a new op trace entry.
    void write(const OpTraceEntry &entry) {
        size_t pos = write_pos_.load(std::memory_order_relaxed);
        entries_[pos % capacity_] = entry;
        write_pos_.store(pos + 1, std::memory_order_release);
        total_count_.fetch_add(1, std::memory_order_relaxed);
    }

    /// Get the most recent N entries (ordered oldest to newest).
    /// Returns fewer entries if the ring hasn't filled up yet.
    std::vector<OpTraceEntry> getRecentEntries(size_t n) const {
        size_t wp = write_pos_.load(std::memory_order_acquire);
        size_t available = wp < capacity_ ? wp : capacity_;
        size_t count = n < available ? n : available;

        std::vector<OpTraceEntry> result;
        result.reserve(count);

        // Read from (wp - count) to (wp - 1)
        for (size_t i = wp - count; i < wp; ++i) {
            result.push_back(entries_[i % capacity_]);
        }
        return result;
    }

    /// Get all valid entries in the ring (ordered oldest to newest).
    std::vector<OpTraceEntry> getAllEntries() const {
        return getRecentEntries(capacity_);
    }

    /// Total number of ops traced since creation.
    size_t totalCount() const {
        return total_count_.load(std::memory_order_relaxed);
    }

    /// Current number of valid entries in the ring.
    size_t size() const {
        size_t wp = write_pos_.load(std::memory_order_relaxed);
        return wp < capacity_ ? wp : capacity_;
    }

    /// Ring capacity.
    size_t capacity() const { return capacity_; }

    /// Clear all entries.
    void clear() {
        write_pos_.store(0, std::memory_order_relaxed);
        total_count_.store(0, std::memory_order_relaxed);
    }

private:
    size_t capacity_;
    std::vector<OpTraceEntry> entries_;
    std::atomic<size_t> write_pos_;
    std::atomic<size_t> total_count_;
};

// ============================================================
// Global OpTrace singleton access
// ============================================================

/// Get the global OpTraceRing instance.
/// This is the primary entry point for recording op traces.
OpTraceRing &getGlobalOpTrace();

/// Record an op invocation to the global trace ring.
/// This is the function called from the INFINICORE_GRAPH_OP_RECORD_OR_RUN
/// macro hook (when ENABLE_MUTUAL_AWARENESS is defined).
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

} // namespace infinicore::analyzer
