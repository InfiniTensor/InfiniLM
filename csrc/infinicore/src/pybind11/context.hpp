#pragma once

#include <pybind11/pybind11.h>

#include <cstdint>
#include <string>

#include "infinicore.hpp"

namespace py = pybind11;

namespace infinicore::context {

inline void bind(py::module &m) {
    // Device management
    m.def("get_device", &getDevice, "Get the current active device");
    m.def("get_device_count", &getDeviceCount,
          "Get the number of available devices of a specific type",
          py::arg("device_type"));
    m.def("set_device", &setDevice,
          "Set the current active device",
          py::arg("device"));

    // Stream and handle management
    m.def(
        "get_stream", []() { return reinterpret_cast<std::uintptr_t>(getStream()); },
        "Get the current stream as an integer handle");

    // Synchronization
    m.def("sync_stream", &syncStream, "Synchronize the current stream");
    m.def("sync_device", &syncDevice, "Synchronize the current device");

    // Graph
    m.def("is_graph_recording", &isGraphRecording, "Check if graph recording is turned on");
    m.def("start_graph_recording", &startGraphRecording, "Start graph recording");
    m.def("stop_graph_recording", &stopGraphRecording, "Stop graph recording and return the graph");
    m.def("cancel_graph_recording", &cancelGraphRecording, "Cancel the active graph recording");
    m.def("fa_in_graph_allowed",
          &faInGraphAllowed,
          "Phase-adaptive FA under stream capture (INFINI_FA_INGRAPH)");
    m.def("moe_triton_capture_allowed",
          &moeTritonCaptureAllowed,
          "Phase-adaptive MoE under stream capture (INFINI_MOE_INGRAPH)");
    m.def(
        "set_inference_phase",
        [](const std::string &phase) {
            if (phase == "decode" || phase == "Decode") {
                setInferencePhase(InferencePhase::Decode);
            } else if (phase == "prefill" || phase == "Prefill") {
                setInferencePhase(InferencePhase::Prefill);
            } else {
                setInferencePhase(InferencePhase::Unknown);
            }
        },
        py::arg("phase"),
        "Set TLS InferencePhase: decode | prefill | unknown");
    m.def(
        "get_inference_phase",
        []() -> std::string {
            switch (getInferencePhase()) {
            case InferencePhase::Decode:
                return "decode";
            case InferencePhase::Prefill:
                return "prefill";
            default:
                return "unknown";
            }
        },
        "Current TLS InferencePhase string");
}

} // namespace infinicore::context
