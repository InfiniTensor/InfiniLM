#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/all.hpp"

namespace py = pybind11;

namespace infinicore::ops {

Tensor py_all(Tensor input, py::object dim, bool keepdim) {
    if (dim.is_none()) {
        std::vector<size_t> dim_vec;
        for (int i = 0; i < input->shape().size(); i++) {
            dim_vec.push_back(i);
        }
        return op::all(input, dim_vec, keepdim);
    } else if (py::isinstance<py::tuple>(dim) || py::isinstance<py::list>(dim)) {
        return op::all(input, dim.cast<std::vector<size_t>>(), keepdim);
    } else if (py::isinstance<py::int_>(dim)) {
        return op::all(input, std::vector<size_t>(1, dim.cast<size_t>()), keepdim);
    } else {
        throw std::invalid_argument("dim must be a tuple or an integer");
    }
}

void py_all_(Tensor output, Tensor input, py::object dim, bool keepdim) {
    if (dim.is_none()) {
        std::vector<size_t> dim_vec;
        for (int i = 0; i < input->shape().size(); i++) {
            dim_vec.push_back(i);
        }
        op::all_(output, input, dim_vec, keepdim);
    } else if (py::isinstance<py::tuple>(dim) || py::isinstance<py::list>(dim)) {
        op::all_(output, input, dim.cast<std::vector<size_t>>(), keepdim);
    } else if (py::isinstance<py::int_>(dim)) {
        op::all_(output, input, std::vector<size_t>(1, dim.cast<size_t>()), keepdim);
    } else {
        throw std::invalid_argument("dim must be a tuple or an integer");
    }
}

inline void bind_all(py::module &m) {
    m.def("all",
          &py_all,
          py::arg("input"),
          py::arg("dim"),
          py::arg("keepdim"),
          R"doc(All of input tensor along the given dimensions.)doc");

    m.def("all_",
          &py_all_,
          py::arg("output"),
          py::arg("input"),
          py::arg("dim"),
          py::arg("keepdim"),
          R"doc(In-place tensor all.)doc");
}

} // namespace infinicore::ops
