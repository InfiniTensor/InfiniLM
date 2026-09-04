#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/var.hpp"

namespace py = pybind11;

namespace infinicore::ops {

Tensor py_var(Tensor input, py::object dim, bool unbiased, bool keepdim) {
    if (dim.is_none()) {
        std::vector<size_t> dim_vec;
        for (int i = 0; i < input->shape().size(); i++) {
            dim_vec.push_back(i);
        }
        return op::var(input, dim_vec, unbiased, keepdim);
    } else if (py::isinstance<py::tuple>(dim) || py::isinstance<py::list>(dim)) {
        return op::var(input, dim.cast<std::vector<size_t>>(), unbiased, keepdim);
    } else if (py::isinstance<py::int_>(dim)) {
        return op::var(input, std::vector<size_t>(1, dim.cast<size_t>()), unbiased, keepdim);
    } else {
        throw std::invalid_argument("dim must be a tuple or an integer");
    }
}

void py_var_(Tensor var_output, Tensor input, py::object dim, bool unbiased, bool keepdim) {
    if (dim.is_none()) {
        std::vector<size_t> dim_vec;
        for (int i = 0; i < input->shape().size(); i++) {
            dim_vec.push_back(i);
        }
        op::var_(var_output, input, dim_vec, unbiased, keepdim);
    } else if (py::isinstance<py::tuple>(dim) || py::isinstance<py::list>(dim)) {
        op::var_(var_output, input, dim.cast<std::vector<size_t>>(), unbiased, keepdim);
    } else if (py::isinstance<py::int_>(dim)) {
        op::var_(var_output, input, std::vector<size_t>(1, dim.cast<size_t>()), unbiased, keepdim);
    } else {
        throw std::invalid_argument("dim must be a list/tuple or an integer");
    }
}

inline void bind_var(py::module &m) {
    m.def("var",
          &py_var,
          py::arg("input"),
          py::arg("dim"),
          py::arg("unbiased"),
          py::arg("keepdim"),
          R"doc(Var of input tensor along the given dimensions.)doc");

    m.def("var_",
          &py_var_,
          py::arg("var_output"),
          py::arg("input"),
          py::arg("dim"),
          py::arg("unbiased"),
          py::arg("keepdim"),
          R"doc(In-place tensor Var .)doc");
}

} // namespace infinicore::ops
