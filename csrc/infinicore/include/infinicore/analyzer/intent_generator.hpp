#pragma once

#include "op_trace.hpp"
#include "optimization_intent.hpp"

#include <algorithm>
#include <vector>

namespace infinicore::analyzer {

/// IntentGenerator — the core "mutual awareness" logic.
///
/// This is where task demand and resource supply are jointly
/// analyzed to produce an OptimizationIntent. It implements
/// the key insight: the same task phase has different optimization
/// needs under different resource conditions, and the same resource
/// state has different supply value under different task phases.
class IntentGenerator {
public:
    IntentGenerator() = default;

    /// Generate the global semantic intent from phase detection
    /// result and op trace window.
    GlobalSemanticIntent generateGlobal(
        PhaseType phase,
        const std::vector<OpTraceEntry> &window,
        const std::vector<DeviceLocalIntent> &device_intents) const {

        GlobalSemanticIntent intent;
        intent.current_phase = phase;
        intent.timestamp_ns = OpTraceEntry::now();

        if (!window.empty()) {
            intent.op_window_start = 0;
            intent.op_window_end = static_cast<uint32_t>(window.size());
        }

        // --- Compute intensity estimation ---
        intent.compute_intensity = estimateComputeIntensity(phase, window);

        // --- Determine primary bottleneck (mutual awareness) ---
        intent.primary_bottleneck = determineGlobalBottleneck(phase, device_intents);

        // --- Set optimization goal based on phase + bottleneck ---
        intent.goal = determineGoal(phase, intent.primary_bottleneck);

        // --- Generate strategy hints ---
        intent.strategy = generateStrategy(phase, intent.primary_bottleneck, device_intents);

        // --- Confidence ---
        intent.confidence = computeConfidence(phase, window);

        return intent;
    }

    /// Build the complete two-layer OptimizationIntent.
    OptimizationIntent generate(
        PhaseType phase,
        const std::vector<OpTraceEntry> &window,
        const std::vector<DeviceLocalIntent> &device_intents) const {

        OptimizationIntent result;
        result.global = generateGlobal(phase, window, device_intents);
        result.per_device = device_intents;
        return result;
    }

private:
    /// Estimate compute intensity (higher = more compute-heavy).
    /// Uses a simple heuristic based on op type composition.
    float estimateComputeIntensity(
        PhaseType phase,
        const std::vector<OpTraceEntry> &window) const {

        if (window.empty()) {
            return 0.0f;
        }

        size_t heavy_compute_ops = 0;
        for (auto &e : window) {
            if (isGemmMlpOp(e.op_type) || isAttentionOp(e.op_type)) {
                heavy_compute_ops++;
            }
        }
        return static_cast<float>(heavy_compute_ops) / static_cast<float>(window.size());
    }

    /// Determine global bottleneck by jointly considering phase and
    /// per-device resource state (the core mutual awareness logic).
    BottleneckType determineGlobalBottleneck(
        PhaseType phase,
        const std::vector<DeviceLocalIntent> &device_intents) const {

        bool any_memory_bound = false;
        bool any_compute_bound = false;
        bool any_bandwidth_bound = false;
        bool any_communication_bound = false;
        for (auto &d : device_intents) {
            any_memory_bound = any_memory_bound || d.local_bottleneck == BottleneckType::MEMORY_BOUND;
            any_compute_bound = any_compute_bound || d.local_bottleneck == BottleneckType::COMPUTE_BOUND;
            any_bandwidth_bound = any_bandwidth_bound || d.local_bottleneck == BottleneckType::BANDWIDTH_BOUND;
            any_communication_bound = any_communication_bound || d.local_bottleneck == BottleneckType::COMMUNICATION_BOUND;
        }

        // --- Mutual awareness logic ---
        // The same resource state has different "supply value" depending on phase:

        if (any_memory_bound) {
            return BottleneckType::MEMORY_BOUND;
        }

        if (phase == PhaseType::COMMUNICATION || any_communication_bound) {
            return BottleneckType::COMMUNICATION_BOUND;
        }

        switch (phase) {
        case PhaseType::ATTENTION_DENSE:
        case PhaseType::PREFILL:
            // Attention/prefill is dominated by memory movement and KV access,
            // so phase semantics should win unless memory/communication already
            // forced an earlier return above.
            if (any_bandwidth_bound) {
                return BottleneckType::BANDWIDTH_BOUND;
            }
            return BottleneckType::BANDWIDTH_BOUND;

        case PhaseType::GEMM_MLP_DENSE:
            if (any_compute_bound) {
                return BottleneckType::COMPUTE_BOUND;
            }
            if (any_bandwidth_bound) {
                return BottleneckType::BANDWIDTH_BOUND;
            }
            return BottleneckType::COMPUTE_BOUND;

        case PhaseType::DECODE:
            if (any_bandwidth_bound) {
                return BottleneckType::BANDWIDTH_BOUND;
            }
            if (any_compute_bound) {
                return BottleneckType::COMPUTE_BOUND;
            }
            return BottleneckType::BANDWIDTH_BOUND;

        case PhaseType::KV_CACHE:
            if (any_bandwidth_bound) {
                return BottleneckType::BANDWIDTH_BOUND;
            }
            return BottleneckType::MEMORY_BOUND;

        default:
            if (any_bandwidth_bound) {
                return BottleneckType::BANDWIDTH_BOUND;
            }
            if (any_compute_bound) {
                return BottleneckType::COMPUTE_BOUND;
            }
            return BottleneckType::BALANCED;
        }
    }

    /// Determine optimization goal based on phase and bottleneck.
    OptimizationGoal determineGoal(
        PhaseType phase,
        BottleneckType bottleneck) const {

        // Under memory pressure, prioritize memory safety
        if (bottleneck == BottleneckType::MEMORY_BOUND) {
            return OptimizationGoal::MEMORY_SAFE;
        }

        if (bottleneck == BottleneckType::COMMUNICATION_BOUND) {
            return OptimizationGoal::STABILITY_FIRST;
        }

        switch (phase) {
        case PhaseType::DECODE:
            // Decode latency is user-facing → latency first
            return OptimizationGoal::LATENCY_FIRST;

        case PhaseType::PREFILL:
            // Prefill processes a full prompt → throughput first
            return OptimizationGoal::THROUGHPUT_FIRST;

        case PhaseType::ATTENTION_DENSE:
            return OptimizationGoal::LATENCY_FIRST;

        case PhaseType::GEMM_MLP_DENSE:
            return OptimizationGoal::THROUGHPUT_FIRST;

        default:
            return OptimizationGoal::LATENCY_FIRST;
        }
    }

    /// Generate strategy hints from phase + bottleneck + resources.
    StrategyHint generateStrategy(
        PhaseType phase,
        BottleneckType bottleneck,
        const std::vector<DeviceLocalIntent> &device_intents) const {

        StrategyHint hint;

        // Fusion is beneficial for bandwidth-bound phases (reduce memory traffic)
        hint.prefer_fused_ops = (bottleneck == BottleneckType::BANDWIDTH_BOUND)
                             || phase == PhaseType::DECODE;

        // In-place when memory is tight
        hint.prefer_in_place = (bottleneck == BottleneckType::MEMORY_BOUND);

        // Recomputation (activation checkpointing) when memory is critical
        bool extreme_memory = false;
        for (auto &d : device_intents) {
            if (d.memory_usage_ratio >= 0.95f) {
                extreme_memory = true;
                break;
            }
        }
        hint.prefer_recomputation = extreme_memory;

        // Async comm overlap for multi-device and communication phases
        hint.prefer_async_comm = (device_intents.size() > 1)
                              && (phase == PhaseType::GEMM_MLP_DENSE
                                  || phase == PhaseType::COMMUNICATION);

        return hint;
    }

    /// Compute confidence based on how clear the phase signal is.
    float computeConfidence(
        PhaseType phase,
        const std::vector<OpTraceEntry> &window) const {

        if (window.empty() || phase == PhaseType::UNKNOWN) {
            return 0.0f;
        }

        // Count how many ops in the window match the detected phase
        size_t matching = 0;
        for (auto &e : window) {
            bool match = false;
            switch (phase) {
            case PhaseType::ATTENTION_DENSE:
            case PhaseType::PREFILL:
                match = isAttentionOp(e.op_type);
                break;
            case PhaseType::GEMM_MLP_DENSE:
                match = isGemmMlpOp(e.op_type) || isActivationOp(e.op_type);
                break;
            case PhaseType::KV_CACHE:
                match = isKvCacheOp(e.op_type);
                break;
            case PhaseType::DECODE:
                match = isAttentionOp(e.op_type) || isGemmMlpOp(e.op_type);
                break;
            default:
                break;
            }
            if (match) {
                matching++;
            }
        }

        return static_cast<float>(matching) / static_cast<float>(window.size());
    }
};

} // namespace infinicore::analyzer
