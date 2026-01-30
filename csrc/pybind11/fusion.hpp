/**
 * @file fusion.hpp
 * @brief pybind11 bindings for FusionContext
 */

#pragma once

#include "../fusion/fusion_context.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace infinilm::fusion {

inline void bind_fusion(py::module &m) {
    py::class_<FusionContext>(m, "FusionContext")
        .def_static("set", &FusionContext::set,
                    py::arg("op_name"), py::arg("should_fuse"),
                    "Set fusion decision for an operation")
        .def_static("get", &FusionContext::get,
                    py::arg("op_name"), py::arg("default_value") = true,
                    "Get fusion decision for an operation")
        .def_static("has", &FusionContext::has,
                    py::arg("op_name"),
                    "Check if fusion decision is set for an operation")
        .def_static("clear", &FusionContext::clear,
                    "Clear all fusion decisions")
        .def_static("size", &FusionContext::size,
                    "Get number of decisions currently set");
}

} // namespace infinilm::fusion
