#pragma once

#include <pybind11/pybind11.h>

#include <cstdint>

#include "runtime/context.hpp"
#include "runtime/runtime.hpp"

namespace py = pybind11;

namespace infinilm::runtime::context {

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
        "get_stream", []() { return reinterpret_cast<std::uintptr_t>(currentRuntime()->stream()); },
        "Get the current stream as an integer handle");

    // Synchronization
    m.def("sync_stream", []() { currentRuntime()->syncStream(); }, "Synchronize the current stream");
    m.def("sync_device", []() { currentRuntime()->syncDevice(); }, "Synchronize the current device");

}

} // namespace infinilm::runtime::context
