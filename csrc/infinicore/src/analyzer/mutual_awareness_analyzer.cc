#include "infinicore/analyzer/mutual_awareness_analyzer.hpp"

namespace infinicore::analyzer {

namespace {

std::vector<DeviceResourceSnapshot> collectRuntimeResourceSnapshots() {
    std::vector<DeviceResourceSnapshot> device_snapshots;
    // TODO: integrate with ContextImpl allocator stats when available.
    // For now, return an empty snapshot list.
    return device_snapshots;
}

} // namespace

// ============================================================
// Singleton
// ============================================================

MutualAwarenessAnalyzer &MutualAwarenessAnalyzer::instance() {
    static MutualAwarenessAnalyzer inst;
    return inst;
}

MutualAwarenessAnalyzer::MutualAwarenessAnalyzer()
    : phase_detector_(),
      resource_sensor_(),
      intent_generator_(),
      enabled_(true),
      graph_intent_cached_(false) {
}

// ============================================================
// Main analysis entry points
// ============================================================

OptimizationIntent MutualAwarenessAnalyzer::analyze() {
    if (!enabled_) {
        return OptimizationIntent{};
    }

    // If we have a cached graph intent, return it
    if (graph_intent_cached_) {
        return graph_cached_intent_;
    }

    // Get recent op trace window
    auto &trace = getGlobalOpTrace();
    auto window = trace.getRecentEntries(phase_detector_.config().window_size);

    // Detect phase
    PhaseType phase = phase_detector_.detect(window);

    auto device_snapshots = collectRuntimeResourceSnapshots();
    std::vector<DeviceLocalIntent> device_intents;
    device_intents.reserve(device_snapshots.size());
    for (auto const &snapshot : device_snapshots) {
        device_intents.push_back(resource_sensor_.sense(snapshot));
    }

    // Generate intent
    auto intent = intent_generator_.generate(phase, window, device_intents);

    // Cache result
    {
        std::lock_guard<std::mutex> lock(mutex_);
        last_intent_ = intent;
    }

    return intent;
}

OptimizationIntent MutualAwarenessAnalyzer::analyze(
    const std::vector<std::pair<int, MemoryStats>> &device_stats) {

    if (!enabled_) {
        return OptimizationIntent{};
    }

    // If we have a cached graph intent, return it
    if (graph_intent_cached_) {
        return graph_cached_intent_;
    }

    // Get recent op trace window
    auto &trace = getGlobalOpTrace();
    auto window = trace.getRecentEntries(phase_detector_.config().window_size);

    // Detect phase
    PhaseType phase = phase_detector_.detect(window);

    // Build per-device intents from provided stats
    std::vector<DeviceLocalIntent> device_intents;
    device_intents.reserve(device_stats.size());
    for (auto &[dev_id, stats] : device_stats) {
        device_intents.push_back(resource_sensor_.sense(dev_id, stats));
    }

    // Generate intent
    auto intent = intent_generator_.generate(phase, window, device_intents);

    // Cache result
    {
        std::lock_guard<std::mutex> lock(mutex_);
        last_intent_ = intent;
    }

    return intent;
}

OptimizationIntent MutualAwarenessAnalyzer::analyze(
    const std::vector<DeviceResourceSnapshot> &device_snapshots) {

    if (!enabled_) {
        return OptimizationIntent{};
    }

    if (graph_intent_cached_) {
        return graph_cached_intent_;
    }

    auto &trace = getGlobalOpTrace();
    auto window = trace.getRecentEntries(phase_detector_.config().window_size);

    PhaseType phase = phase_detector_.detect(window);

    std::vector<DeviceLocalIntent> device_intents;
    device_intents.reserve(device_snapshots.size());
    for (auto const &snapshot : device_snapshots) {
        device_intents.push_back(resource_sensor_.sense(snapshot));
    }

    auto intent = intent_generator_.generate(phase, window, device_intents);

    {
        std::lock_guard<std::mutex> lock(mutex_);
        last_intent_ = intent;
    }

    return intent;
}

PhaseType MutualAwarenessAnalyzer::getCurrentPhase() const {
    if (!enabled_) {
        return PhaseType::UNKNOWN;
    }

    return phase_detector_.detectFromTrace(getGlobalOpTrace());
}

OptimizationGoal MutualAwarenessAnalyzer::getCurrentOptimizationGoal() const {
    return const_cast<MutualAwarenessAnalyzer *>(this)->analyze().global.goal;
}

const OptimizationIntent &MutualAwarenessAnalyzer::lastIntent() const {
    return last_intent_;
}

// ============================================================
// Graph recording support
// ============================================================

void MutualAwarenessAnalyzer::onGraphRecordingStop() {
    if (!enabled_) {
        return;
    }

    // Analyze the op sequence recorded during graph capture
    // and cache the result. Graph ops are static, so we only
    // need to analyze once.
    graph_cached_intent_ = analyze();
    graph_intent_cached_ = true;
}

void MutualAwarenessAnalyzer::clearGraphCache() {
    graph_intent_cached_ = false;
    graph_cached_intent_ = OptimizationIntent{};
}

// ============================================================
// C-style API for external framework integration
// ============================================================

OptimizationIntent analyzeCurrentState() {
    return MutualAwarenessAnalyzer::instance().analyze();
}

PhaseType getCurrentPhase() {
    return MutualAwarenessAnalyzer::instance().getCurrentPhase();
}

OptimizationGoal getCurrentOptimizationGoal() {
    return MutualAwarenessAnalyzer::instance().getCurrentOptimizationGoal();
}

void setAnalyzerEnabled(bool enabled) {
    MutualAwarenessAnalyzer::instance().setEnabled(enabled);
}

} // namespace infinicore::analyzer
