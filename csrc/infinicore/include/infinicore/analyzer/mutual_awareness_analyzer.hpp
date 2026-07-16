#pragma once

#include "intent_generator.hpp"
#include "op_trace.hpp"
#include "optimization_intent.hpp"
#include "phase_detector.hpp"
#include "resource_sensor.hpp"

#include <mutex>
#include <vector>

namespace infinicore::analyzer {

/// MutualAwarenessAnalyzer — the top-level facade for the
/// hardware-task mutual awareness requirements analysis module.
///
/// This is the primary entry point exposed to external frameworks
/// (e.g., InfiniLM) via C++ function calls. It orchestrates:
///   1. Op trace collection (via OpTraceRing)
///   2. Phase detection (via PhaseDetector)
///   3. Resource sensing (via ResourceSensor)
///   4. Intent generation (via IntentGenerator)
///
/// Usage:
///   auto& analyzer = MutualAwarenessAnalyzer::instance();
///   // ... ops execute and get traced automatically ...
///   auto intent = analyzer.analyze();  // Produces OptimizationIntent
///
/// Thread safety: analyze() is safe to call from any thread.
/// The analyzer reads a snapshot of the op trace ring.
class MutualAwarenessAnalyzer {
public:
    /// Get the singleton instance.
    static MutualAwarenessAnalyzer &instance();

    // Non-copyable, non-movable
    MutualAwarenessAnalyzer(const MutualAwarenessAnalyzer &) = delete;
    MutualAwarenessAnalyzer &operator=(const MutualAwarenessAnalyzer &) = delete;

    /// Main analysis entry point.
    /// Analyzes the current op trace window + resource state
    /// and returns a complete OptimizationIntent.
    ///
    /// This is the function InfiniLM should call.
    /// Latency: expected < 1ms for MVP rule-based analysis.
    OptimizationIntent analyze();

    /// Analyze with explicitly provided memory stats per device.
    /// Use this when the caller can provide resource info directly.
    OptimizationIntent analyze(const std::vector<std::pair<int, MemoryStats>> &device_stats);

    /// Analyze with explicitly provided device resource snapshots.
    /// This is the richer input path used by demand-analysis-oriented callers.
    OptimizationIntent analyze(const std::vector<DeviceResourceSnapshot> &device_snapshots);

    /// Get the current phase without generating full intent.
    /// Lightweight query for simple use cases.
    PhaseType getCurrentPhase() const;

    /// Get the current optimization goal derived from the
    /// latest analyzer result.
    OptimizationGoal getCurrentOptimizationGoal() const;

    /// Get the most recent OptimizationIntent (cached from last analyze()).
    const OptimizationIntent &lastIntent() const;

    /// Access the underlying components for configuration.
    PhaseDetector &phaseDetector() { return phase_detector_; }
    ResourceSensor &resourceSensor() { return resource_sensor_; }
    OpTraceRing &opTrace() { return getGlobalOpTrace(); }

    /// Enable / disable the analyzer.
    /// When disabled, analyze() returns a default intent and
    /// op trace recording is skipped.
    void setEnabled(bool enabled) { enabled_ = enabled; }
    bool isEnabled() const { return enabled_; }

    /// Graph recording support: when graph recording stops,
    /// analyze the recorded op sequence once and cache the result.
    /// Subsequent calls return the cached intent without re-analysis.
    void onGraphRecordingStop();
    void clearGraphCache();

private:
    MutualAwarenessAnalyzer();

    PhaseDetector phase_detector_;
    ResourceSensor resource_sensor_;
    IntentGenerator intent_generator_;

    OptimizationIntent last_intent_;
    mutable std::mutex mutex_;

    bool enabled_ = true;

    // Graph recording cache
    bool graph_intent_cached_ = false;
    OptimizationIntent graph_cached_intent_;
};

// ============================================================
// C-style API for external framework integration (e.g., InfiniLM)
// ============================================================

/// Analyze current state and return an OptimizationIntent.
/// This is the simplest API for external frameworks to call.
OptimizationIntent analyzeCurrentState();

/// Get the current detected phase.
PhaseType getCurrentPhase();

/// Get the current optimization goal.
OptimizationGoal getCurrentOptimizationGoal();

/// Enable / disable the mutual awareness analyzer.
void setAnalyzerEnabled(bool enabled);

} // namespace infinicore::analyzer
