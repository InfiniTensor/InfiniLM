#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // 添加这行

#include "infinicore/ops/topk.hpp"

namespace py = pybind11;

namespace infinicore::ops {

std::pair<Tensor, Tensor> py_topk(Tensor input, size_t k, int dim, bool largest, bool sorted) {
    if (dim == -1) {
        return op::topk(input, k, input->ndim() - 1, largest, sorted);
    } else if (dim >= 0) {
        return op::topk(input, k, static_cast<size_t>(dim), largest, sorted);
    } else {
        throw std::invalid_argument("invalid argument: dim");
    }
}

void py_topk_(Tensor values_output, Tensor indices_output, Tensor input, size_t k, int dim, bool largest, bool sorted) {
    if (dim == -1) {
        op::topk_(values_output, indices_output, input, k, input->ndim() - 1, largest, sorted);
    } else if (dim >= 0) {
        op::topk_(values_output, indices_output, input, k, static_cast<size_t>(dim), largest, sorted);
    } else {
        throw std::invalid_argument("invalid argument: dim");
    }
}

inline void bind_topk(py::module &m) {
    m.def("topk",
          &py_topk,
          py::arg("input"),
          py::arg("k"),
          py::arg("dim"),
          py::arg("largest"),
          py::arg("sorted"),
          R"doc(topk of input tensor along the given dimensions.)doc");

    m.def("topk_",
          &py_topk_,
          py::arg("values_output"),
          py::arg("indices_output"),
          py::arg("input"),
          py::arg("k"),
          py::arg("dim"),
          py::arg("largest"),
          py::arg("sorted"),
          R"doc(In-place tensor topk_.)doc");
}

} // namespace infinicore::ops
