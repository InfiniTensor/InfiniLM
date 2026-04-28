#pragma once

#include <pybind11/pybind11.h>

#include "infinicore.hpp"

namespace py = pybind11;

namespace infinicore::device {

inline void bind(py::module &m) {
    py::class_<Device> device(m, "Device");

    py::enum_<Device::Type>(device, "Type")
        .value("CPU", Device::Type::CPU)
        .value("NVIDIA", Device::Type::NVIDIA)
        .value("CAMBRICON", Device::Type::CAMBRICON)
        .value("ASCEND", Device::Type::ASCEND)
        .value("METAX", Device::Type::METAX)
        .value("MOORE", Device::Type::MOORE)
        .value("ILUVATAR", Device::Type::ILUVATAR)
        .value("QY", Device::Type::QY)
        .value("KUNLUN", Device::Type::KUNLUN)
        .value("HYGON", Device::Type::HYGON)
        .value("ALI", Device::Type::ALI)
        .value("COUNT", Device::Type::COUNT);

    device
        .def(py::init<const Device::Type &, const Device::Index &>(),
             py::arg("type") = Device::Type::CPU, py::arg("index") = 0)
        .def_property_readonly("type", &Device::getType)
        .def_property_readonly("index", &Device::getIndex)
        .def("__str__", static_cast<std::string (Device::*)() const>(&Device::toString));
}

} // namespace infinicore::device
