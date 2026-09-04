#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/pad.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_pad(py::module &m) {
    m.def("pad",
          &op::pad,
          py::arg("x"),
          py::arg("pad"),
          py::arg("mode") = std::string("constant"),
          py::arg("value") = 0.0,
          R"doc(Pad a tensor (PyTorch padding order).)doc");

    m.def("pad_",
          &op::pad_,
          py::arg("y"),
          py::arg("x"),
          py::arg("pad"),
          py::arg("mode") = std::string("constant"),
          py::arg("value") = 0.0,
          R"doc(Out variant of pad.)doc");
}

} // namespace infinicore::ops
