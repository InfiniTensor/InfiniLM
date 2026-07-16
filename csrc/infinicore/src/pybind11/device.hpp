#pragma once

#include <pybind11/pybind11.h>

#include "infinicore.hpp"

namespace py = pybind11;

namespace infinicore::device {

inline void bind(py::module &m) {
    py::class_<Device> device(m, "Device");

    py::enum_<Device::Type>(device, "Type")
        .value("CPU", Device::Type::kCpu)
        .value("NVIDIA", Device::Type::kNvidia)
        .value("CAMBRICON", Device::Type::kCambricon)
        .value("ASCEND", Device::Type::kAscend)
        .value("METAX", Device::Type::kMetax)
        .value("MOORE", Device::Type::kMoore)
        .value("ILUVATAR", Device::Type::kIluvatar)
        .value("HYGON", Device::Type::kHygon);

    device
        .def(py::init<const Device::Type &, const int &>(),
             py::arg("type") = Device::Type::kCpu, py::arg("index") = 0)
        .def_property_readonly("type", &Device::type)
        .def_property_readonly("index", &Device::index)
        .def("__str__", &Device::ToString);
}

} // namespace infinicore::device
