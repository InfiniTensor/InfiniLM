#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/adaptive_avg_pool3d.hpp"

namespace py = pybind11;
namespace infinicore::ops {
inline void bind_adaptive_avg_pool3d(py::module &m) {
    m.def("adaptive_avg_pool3d",
          &op::adaptive_avg_pool3d,
          py::arg("x"),
          py::arg("output_size"),
          R"doc( Adaptive Average Pooling 3D.)doc");

    m.def("adaptive_avg_pool3d_",
          &op::adaptive_avg_pool3d_,
          py::arg("y"),
          py::arg("x"),
          R"doc(In-place, Adaptive Average Pooling 3D.)doc");
}
} // namespace infinicore::ops
