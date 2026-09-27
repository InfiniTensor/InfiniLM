#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/fmod.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_fmod(py::module &m) {
    m.def("fmod",
          &op::fmod,
          py::arg("a"),
          py::arg("b"),
          R"doc(Element-wise floating point remainder of division of two tensors.)doc");

    m.def("fmod_",
          &op::fmod_,
          py::arg("c"),
          py::arg("a"),
          py::arg("b"),
          R"doc(In-place element-wise floating point remainder of division of two tensors.)doc");
}

} // namespace infinicore::ops
