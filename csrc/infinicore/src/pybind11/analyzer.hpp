#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "infinicore/analyzer/mutual_awareness_analyzer.hpp"

namespace py = pybind11;

namespace infinicore::analyzer::pybind {

inline void bind(py::module &m) {
    auto analyzer_mod = m.def_submodule("analyzer",
                                        "Hardware-Task Mutual Awareness Analysis Module");

    // --- Enums ---
    py::enum_<PhaseType>(analyzer_mod, "PhaseType")
        .value("UNKNOWN", PhaseType::UNKNOWN)
        .value("PREFILL", PhaseType::PREFILL)
        .value("DECODE", PhaseType::DECODE)
        .value("ATTENTION_DENSE", PhaseType::ATTENTION_DENSE)
        .value("GEMM_MLP_DENSE", PhaseType::GEMM_MLP_DENSE)
        .value("MOE_ROUTING", PhaseType::MOE_ROUTING)
        .value("KV_CACHE", PhaseType::KV_CACHE)
        .value("COMMUNICATION", PhaseType::COMMUNICATION)
        .export_values();

    py::enum_<BottleneckType>(analyzer_mod, "BottleneckType")
        .value("COMPUTE_BOUND", BottleneckType::COMPUTE_BOUND)
        .value("MEMORY_BOUND", BottleneckType::MEMORY_BOUND)
        .value("BANDWIDTH_BOUND", BottleneckType::BANDWIDTH_BOUND)
        .value("COMMUNICATION_BOUND", BottleneckType::COMMUNICATION_BOUND)
        .value("BALANCED", BottleneckType::BALANCED)
        .export_values();

    py::enum_<OptimizationGoal>(analyzer_mod, "OptimizationGoal")
        .value("LATENCY_FIRST", OptimizationGoal::LATENCY_FIRST)
        .value("THROUGHPUT_FIRST", OptimizationGoal::THROUGHPUT_FIRST)
        .value("MEMORY_SAFE", OptimizationGoal::MEMORY_SAFE)
        .value("STABILITY_FIRST", OptimizationGoal::STABILITY_FIRST)
        .export_values();

    py::enum_<OpType>(analyzer_mod, "OpType")
        .value("UNKNOWN", OpType::UNKNOWN)
        .value("ATTENTION", OpType::ATTENTION)
        .value("FLASH_ATTENTION", OpType::FLASH_ATTENTION)
        .value("GEMM", OpType::GEMM)
        .value("LINEAR", OpType::LINEAR)
        .value("MATMUL", OpType::MATMUL)
        .value("SILU", OpType::SILU)
        .value("GELU", OpType::GELU)
        .value("RMS_NORM", OpType::RMS_NORM)
        .value("KV_CACHING", OpType::KV_CACHING)
        .value("PAGED_CACHING", OpType::PAGED_CACHING)
        .value("EMBEDDING", OpType::EMBEDDING)
        .value("ROPE", OpType::ROPE)
        .export_values();

    // --- StrategyHint ---
    py::class_<StrategyHint>(analyzer_mod, "StrategyHint")
        .def(py::init<>())
        .def_readwrite("prefer_fused_ops", &StrategyHint::prefer_fused_ops)
        .def_readwrite("prefer_in_place", &StrategyHint::prefer_in_place)
        .def_readwrite("prefer_recomputation", &StrategyHint::prefer_recomputation)
        .def_readwrite("prefer_async_comm", &StrategyHint::prefer_async_comm)
        .def("__repr__", [](const StrategyHint &s) {
            return "<StrategyHint fused=" + std::to_string(s.prefer_fused_ops)
                 + " inplace=" + std::to_string(s.prefer_in_place)
                 + " recomp=" + std::to_string(s.prefer_recomputation)
                 + " async_comm=" + std::to_string(s.prefer_async_comm) + ">";
        });

    // --- GlobalSemanticIntent ---
    py::class_<GlobalSemanticIntent>(analyzer_mod, "GlobalSemanticIntent")
        .def(py::init<>())
        .def_readwrite("current_phase", &GlobalSemanticIntent::current_phase)
        .def_readwrite("primary_bottleneck", &GlobalSemanticIntent::primary_bottleneck)
        .def_readwrite("goal", &GlobalSemanticIntent::goal)
        .def_readwrite("compute_intensity", &GlobalSemanticIntent::compute_intensity)
        .def_readwrite("confidence", &GlobalSemanticIntent::confidence)
        .def_readwrite("strategy", &GlobalSemanticIntent::strategy)
        .def_readwrite("timestamp_ns", &GlobalSemanticIntent::timestamp_ns)
        .def("__repr__", [](const GlobalSemanticIntent &i) {
            return "<GlobalSemanticIntent phase="
                 + std::string(phaseTypeToString(i.current_phase))
                 + " bottleneck=" + bottleneckTypeToString(i.primary_bottleneck)
                 + " goal=" + optimizationGoalToString(i.goal)
                 + " confidence=" + std::to_string(i.confidence) + ">";
        });

    // --- DeviceLocalIntent ---
    py::class_<DeviceLocalIntent>(analyzer_mod, "DeviceLocalIntent")
        .def(py::init<>())
        .def_readwrite("device_id", &DeviceLocalIntent::device_id)
        .def_readwrite("memory_usage_ratio", &DeviceLocalIntent::memory_usage_ratio)
        .def_readwrite("memory_available_bytes", &DeviceLocalIntent::memory_available_bytes)
        .def_readwrite("local_bottleneck", &DeviceLocalIntent::local_bottleneck)
        .def_readwrite("compute_utilization", &DeviceLocalIntent::compute_utilization)
        .def_readwrite("memory_bandwidth_utilization", &DeviceLocalIntent::memory_bandwidth_utilization)
        .def_readwrite("communication_time_ratio", &DeviceLocalIntent::communication_time_ratio)
        .def_readwrite("resource_confidence", &DeviceLocalIntent::resource_confidence)
        .def("__repr__", [](const DeviceLocalIntent &d) {
            return "<DeviceLocalIntent dev=" + std::to_string(d.device_id)
                 + " mem=" + std::to_string(d.memory_usage_ratio)
                 + " bottleneck=" + bottleneckTypeToString(d.local_bottleneck) + ">";
        });

    // --- OptimizationIntent ---
    py::class_<OptimizationIntent>(analyzer_mod, "OptimizationIntent")
        .def(py::init<>())
        .def_readwrite("global_intent", &OptimizationIntent::global)
        .def_readwrite("per_device", &OptimizationIntent::per_device)
        .def("get_device_intent", &OptimizationIntent::getDeviceIntent,
             py::return_value_policy::reference,
             py::arg("device_id"))
        .def("__repr__", [](const OptimizationIntent &i) {
            return "<OptimizationIntent phase="
                 + std::string(phaseTypeToString(i.global.current_phase))
                 + " devices=" + std::to_string(i.per_device.size()) + ">";
        });

    // --- Top-level functions ---
    analyzer_mod.def("analyze", &analyzeCurrentState,
                     "Analyze current state and return an OptimizationIntent");
    analyzer_mod.def("get_current_phase", &getCurrentPhase,
                     "Get the current detected task phase");
    analyzer_mod.def("set_enabled", &setAnalyzerEnabled,
                     "Enable/disable the mutual awareness analyzer",
                     py::arg("enabled"));
    analyzer_mod.def(
        "trace_op_for_test",
        [](OpType op_type,
           const std::vector<size_t> &shape,
           uint8_t dtype,
           uint8_t device_type,
           int device_id) {
            traceOp(op_type, shape.data(), shape.size(), dtype, device_type, static_cast<int8_t>(device_id));
        },
        "Inject an OpTrace entry for testing",
        py::arg("op_type"),
        py::arg("shape"),
        py::arg("dtype") = 0,
        py::arg("device_type") = 0,
        py::arg("device_id") = 0);
    analyzer_mod.def(
        "clear_trace", []() {
            getGlobalOpTrace().clear();
            MutualAwarenessAnalyzer::instance().clearGraphCache();
        },
        "Clear the global OpTrace ring and analyzer graph cache");

    // --- Access to analyzer instance for advanced usage ---
    analyzer_mod.def("get_analyzer", &MutualAwarenessAnalyzer::instance,
                     py::return_value_policy::reference,
                     "Get the MutualAwarenessAnalyzer singleton instance");

    py::class_<MutualAwarenessAnalyzer>(analyzer_mod, "MutualAwarenessAnalyzer")
        .def("analyze", py::overload_cast<>(&MutualAwarenessAnalyzer::analyze))
        .def("get_current_phase", &MutualAwarenessAnalyzer::getCurrentPhase)
        .def("last_intent", &MutualAwarenessAnalyzer::lastIntent,
             py::return_value_policy::reference)
        .def("set_enabled", &MutualAwarenessAnalyzer::setEnabled)
        .def("is_enabled", &MutualAwarenessAnalyzer::isEnabled)
        .def("on_graph_recording_stop", &MutualAwarenessAnalyzer::onGraphRecordingStop)
        .def("clear_graph_cache", &MutualAwarenessAnalyzer::clearGraphCache);
}

} // namespace infinicore::analyzer::pybind
