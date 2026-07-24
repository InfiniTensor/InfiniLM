#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/var_mean.hpp"

namespace py = pybind11;

namespace infinicore::ops {

std::pair<Tensor, Tensor> py_var_mean(Tensor input, py::object dim, bool unbiased, bool keepdim) {
    if (dim.is_none()) {
        std::vector<size_t> dim_vec;
        for (int i = 0; i < input->shape().size(); i++) {
            dim_vec.push_back(i);
        }
        return op::var_mean(input, dim_vec, unbiased, keepdim);
    } else if (py::isinstance<py::tuple>(dim) || py::isinstance<py::list>(dim)) {
        return op::var_mean(input, dim.cast<std::vector<size_t>>(), unbiased, keepdim);
    } else if (py::isinstance<py::int_>(dim)) {
        return op::var_mean(input, std::vector<size_t>(1, dim.cast<size_t>()), unbiased, keepdim);
    } else {
        throw std::invalid_argument("dim must be a tuple or an integer");
    }
}

void py_var_mean_(Tensor var_output, Tensor mean_output, Tensor input, py::object dim, bool unbiased, bool keepdim) {
    if (dim.is_none()) {
        std::vector<size_t> dim_vec;
        for (int i = 0; i < input->shape().size(); i++) {
            dim_vec.push_back(i);
        }
        op::var_mean_(var_output, mean_output, input, dim_vec, unbiased, keepdim);
    } else if (py::isinstance<py::tuple>(dim) || py::isinstance<py::list>(dim)) {
        op::var_mean_(var_output, mean_output, input, dim.cast<std::vector<size_t>>(), unbiased, keepdim);
    } else if (py::isinstance<py::int_>(dim)) {
        op::var_mean_(var_output, mean_output, input, std::vector<size_t>(1, dim.cast<size_t>()), unbiased, keepdim);
    } else {
        throw std::invalid_argument("dim must be a list/tuple or an integer");
    }
}

inline void bind_var_mean(py::module &m) {
    m.def("var_mean",
          &py_var_mean,
          py::arg("input"),
          py::arg("dim"),
          py::arg("unbiased"),
          py::arg("keepdim"),
          R"doc(Var & Mean of input tensor along the given dimensions.)doc");

    m.def("var_mean_",
          &py_var_mean_,
          py::arg("var_output"),
          py::arg("mean_output"),
          py::arg("input"),
          py::arg("dim"),
          py::arg("unbiased"),
          py::arg("keepdim"),
          R"doc(In-place tensor Var & Mean .)doc");
}

} // namespace infinicore::ops
