#pragma once

#include "infinicore/ops/upsample_nearest.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_upsample_nearest(py::module &m) {
    m.def("upsample_nearest",
          &op::upsample_nearest,
          py::arg("input"),
          py::arg("output_size"),
          R"doc(Upsample the input using nearest neighbor interpolation.

    Args:
        input (Tensor): The input tensor.
        output_size (List[int]): The output spatial size (e.g. [H_out, W_out]).
    )doc");

    m.def("upsample_nearest_",
          &op::upsample_nearest_,
          py::arg("output"),
          py::arg("input"),
          R"doc(Explicit output UpsampleNearest operation. Writes the result into the output tensor.)doc");
}

} // namespace infinicore::ops
