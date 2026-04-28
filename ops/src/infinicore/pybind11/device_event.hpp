#pragma once

#include "infinicore.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::device_event {

inline void bind(py::module &m) {
    py::class_<DeviceEvent>(m, "DeviceEvent")
        .def(py::init<>(), "Construct a DeviceEvent on the current device")
        .def(py::init<uint32_t>(), "Construct a DeviceEvent with specific flags", py::arg("flags"))
        .def(py::init<Device>(), "Construct a DeviceEvent on a specific device", py::arg("device"))
        .def(py::init<Device, uint32_t>(), "Construct a DeviceEvent on a specific device with flags",
             py::arg("device"), py::arg("flags"))

        .def("record", py::overload_cast<>(&DeviceEvent::record),
             "Record the event on the current stream of its device")
        .def("record", py::overload_cast<infinirtStream_t>(&DeviceEvent::record),
             "Record the event on a specific stream", py::arg("stream"))

        .def("synchronize", &DeviceEvent::synchronize,
             "Wait for the event to complete (blocking)")
        .def("query", &DeviceEvent::query,
             "Check if the event has been completed")

        .def("elapsed_time", &DeviceEvent::elapsed_time,
             "Calculate elapsed time between this event and another event (in milliseconds)",
             py::arg("other"))

        .def("wait", &DeviceEvent::wait,
             "Make a stream wait for this event to complete",
             py::arg("stream") = nullptr)

        .def_property_readonly("device", &DeviceEvent::device,
                               "Get the device where this event was created")
        .def_property_readonly("is_recorded", &DeviceEvent::is_recorded,
                               "Check if the event has been recorded");
}

} // namespace infinicore::device_event
