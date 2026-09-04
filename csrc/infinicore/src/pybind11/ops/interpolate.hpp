#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "infinicore/ops/interpolate.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_interpolate(py::module &m) {
    m.def("interpolate",
          &op::interpolate,
          py::arg("input"),
          py::arg("mode"),
          py::arg("size"),
          py::arg("scale_factor"),
          py::arg("align_corners"),
          R"doc(Interpolate (upsample/downsample) a tensor.)doc");

    m.def("interpolate_",
          &op::interpolate_,
          py::arg("out"),
          py::arg("input"),
          py::arg("mode"),
          py::arg("size"),
          py::arg("scale_factor"),
          py::arg("align_corners"),
          R"doc(In-place interpolate (writes to out).)doc");
}

} // namespace infinicore::ops
