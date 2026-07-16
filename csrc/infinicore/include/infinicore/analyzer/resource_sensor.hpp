#pragma once

#include "infinicore/device.hpp"
#include "optimization_intent.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace infinicore::analyzer {

/// Memory statistics from the allocator.
struct MemoryStats {
    size_t allocated_bytes = 0;  // Currently allocated bytes
    size_t total_capacity = 0;   // Total pool capacity in bytes
    size_t peak_allocated = 0;   // Peak allocation since last reset
    size_t allocation_count = 0; // Number of active allocations

    float usageRatio() const {
        return total_capacity > 0
                 ? static_cast<float>(allocated_bytes) / static_cast<float>(total_capacity)
                 : 0.0f;
    }
};

/// A normalized resource image consumed by the analyzer.
///
/// This is intentionally vendor-neutral. Backend-specific runtimes
/// populate these fields and the analyzer consumes only this normalized view.
struct DeviceResourceSnapshot {
    int device_id = -1;
    Device::Type device_type = Device::Type::kCpu;

    bool has_memory_capacity = false;
    bool has_compute_utilization = false;
    bool has_memory_bandwidth_utilization = false;
    bool has_kernel_time_ratio = false;
    bool has_communication = false;
    bool kernel_time_estimated = false;

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    size_t used_bytes = 0;
    size_t reserved_bytes = 0;

    float compute_utilization = 0.0f;
    float memory_bandwidth_utilization = 0.0f;

    uint64_t bytes_read = 0;
    uint64_t bytes_written = 0;

    float kernel_time_ratio = 0.0f;
    float idle_time_ratio = 0.0f;

    float communication_time_ratio = 0.0f;
    uint64_t communication_bytes = 0;

    float load_imbalance_score = 0.0f;

    float memoryUsageRatio() const {
        if (total_bytes > 0) {
            return static_cast<float>(used_bytes) / static_cast<float>(total_bytes);
        }
        if (reserved_bytes > 0 && used_bytes <= reserved_bytes) {
            return static_cast<float>(used_bytes) / static_cast<float>(reserved_bytes);
        }
        return 0.0f;
    }

    float resourceConfidence() const {
        float confidence = 0.0f;
        if (has_memory_capacity) {
            confidence += 0.35f;
        }
        if (has_compute_utilization) {
            confidence += 0.25f;
        }
        if (has_memory_bandwidth_utilization) {
            confidence += 0.25f;
        }
        if (has_communication) {
            confidence += 0.15f;
        }
        return std::min(confidence, 1.0f);
    }
};

/// ResourceSensor — gathers current resource state from
/// the runtime and allocator subsystems.
///
/// This is the "resource supply" side of the mutual-awareness
/// equation. It aggregates device type, memory stats, and
/// potentially timing info into a resource snapshot.
class ResourceSensor {
public:
    ResourceSensor() = default;

    /// Build a DeviceLocalIntent from a normalized resource image.
    DeviceLocalIntent sense(const DeviceResourceSnapshot &snapshot) const {
        DeviceLocalIntent intent;
        intent.device_id = snapshot.device_id;
        intent.memory_usage_ratio = snapshot.memoryUsageRatio();
        intent.memory_available_bytes = snapshot.free_bytes > 0
                                          ? snapshot.free_bytes
                                          : ((snapshot.total_bytes >= snapshot.used_bytes) ? (snapshot.total_bytes - snapshot.used_bytes) : 0);
        intent.compute_utilization = snapshot.compute_utilization;
        intent.memory_bandwidth_utilization = snapshot.memory_bandwidth_utilization;
        intent.communication_time_ratio = snapshot.communication_time_ratio;
        intent.resource_confidence = snapshot.resourceConfidence();

        if (snapshot.has_communication && snapshot.communication_time_ratio > high_communication_threshold_) {
            intent.local_bottleneck = BottleneckType::COMMUNICATION_BOUND;
            return intent;
        }

        if (intent.memory_usage_ratio > high_memory_threshold_) {
            intent.local_bottleneck = BottleneckType::MEMORY_BOUND;
            return intent;
        }

        if (snapshot.has_memory_bandwidth_utilization
            && snapshot.memory_bandwidth_utilization > high_bandwidth_threshold_
            && (!snapshot.has_compute_utilization
                || snapshot.memory_bandwidth_utilization >= snapshot.compute_utilization + bandwidth_margin_)) {
            intent.local_bottleneck = BottleneckType::BANDWIDTH_BOUND;
            return intent;
        }

        if (snapshot.has_compute_utilization && snapshot.compute_utilization > high_compute_threshold_) {
            intent.local_bottleneck = BottleneckType::COMPUTE_BOUND;
            return intent;
        }

        if (intent.memory_usage_ratio > moderate_memory_threshold_) {
            intent.local_bottleneck = BottleneckType::BALANCED;
            return intent;
        }

        // Preserve the MVP fallback: if all we know is that memory is low,
        // compute is the most likely bottleneck.
        intent.local_bottleneck = BottleneckType::COMPUTE_BOUND;
        return intent;
    }

    /// Build a DeviceLocalIntent from current resource state.
    /// In MVP, this primarily queries allocator memory stats.
    ///
    /// @param device_id  The device ID to query
    /// @param stats      Memory stats from the allocator
    DeviceLocalIntent sense(
        int device_id,
        const MemoryStats &stats,
        Device::Type device_type = Device::Type::kCpu) const {
        DeviceResourceSnapshot snapshot;
        snapshot.device_id = device_id;
        snapshot.device_type = device_type;
        snapshot.has_memory_capacity = stats.total_capacity > 0;
        snapshot.free_bytes = stats.total_capacity >= stats.allocated_bytes
                                ? (stats.total_capacity - stats.allocated_bytes)
                                : 0;
        snapshot.total_bytes = stats.total_capacity;
        snapshot.used_bytes = stats.allocated_bytes;
        snapshot.reserved_bytes = stats.total_capacity;
        return sense(snapshot);
    }

    /// Thresholds for memory-based bottleneck classification.
    void setHighMemoryThreshold(float t) { high_memory_threshold_ = t; }
    void setModerateMemoryThreshold(float t) { moderate_memory_threshold_ = t; }

private:
    float high_memory_threshold_ = 0.85f;
    float moderate_memory_threshold_ = 0.5f;
    float high_compute_threshold_ = 0.75f;
    float high_bandwidth_threshold_ = 0.75f;
    float high_communication_threshold_ = 0.25f;
    float bandwidth_margin_ = 0.05f;
};

} // namespace infinicore::analyzer
