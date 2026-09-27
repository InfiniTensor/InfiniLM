#pragma once

#include "infinicore/ops/conv2d.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::ops {

Tensor py_conv2d(Tensor input,
                 Tensor weight,
                 pybind11::object bias,
                 const std::vector<size_t> &padding,
                 const std::vector<size_t> &stride,
                 const std::vector<size_t> &dilation) {
    Tensor bias_tensor;
    if (!bias.is_none()) {
        bias_tensor = bias.cast<Tensor>();
    }
    return op::conv2d(input, weight, bias_tensor, padding, stride, dilation);
}

void py_conv2d_(Tensor out,
                Tensor input,
                Tensor weight,
                pybind11::object bias,
                const std::vector<size_t> &padding,
                const std::vector<size_t> &stride,
                const std::vector<size_t> &dilation) {
    Tensor bias_tensor;
    if (!bias.is_none()) {
        bias_tensor = bias.cast<Tensor>();
    }
    op::conv2d_(out, input, weight, bias_tensor, padding, stride, dilation);
}

inline void bind_conv2d(py::module &m) {
    m.def("conv2d",
          &ops::py_conv2d,
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("padding") = std::vector<size_t>{0, 0},
          py::arg("stride") = std::vector<size_t>{1, 1},
          py::arg("dilation") = std::vector<size_t>{1, 1},
          R"doc(Applies a 2D convolution over an input tensor.)doc");

    m.def("conv2d_",
          &ops::py_conv2d_,
          py::arg("out"),
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("padding") = std::vector<size_t>{0, 0},
          py::arg("stride") = std::vector<size_t>{1, 1},
          py::arg("dilation") = std::vector<size_t>{1, 1},
          R"doc(In-place 2D convolution.)doc");
}

} // namespace infinicore::ops
