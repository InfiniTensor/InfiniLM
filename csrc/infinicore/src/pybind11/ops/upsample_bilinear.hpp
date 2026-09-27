#pragma once

#include "infinicore/ops/upsample_bilinear.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_upsample_bilinear(py::module &m) {
    m.def("upsample_bilinear",
          &op::upsample_bilinear,
          py::arg("input"),
          py::arg("output_size"),
          py::arg("align_corners") = false,
          R"doc(Upsample the input using bilinear interpolation.

    Args:
        input (Tensor): The input tensor.
        output_size (List[int]): The output spatial size (e.g. [H_out, W_out]).
        align_corners (bool): If True, the corner pixels of the input and output tensors are aligned.
    )doc");

    m.def("upsample_bilinear_",
          &op::upsample_bilinear_,
          py::arg("output"),
          py::arg("input"),
          py::arg("align_corners") = false,
          R"doc(Explicit output UpsampleBilinear operation. Writes the result into the output tensor.)doc");
}

} // namespace infinicore::ops
