#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/mul_scalar.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_mul_scalar(py::module &m) {
    m.def("mul_scalar",
          &op::mul_scalar,
          py::arg("a"),
          py::arg("alpha"),
          R"doc(Multiply a tensor by a host scalar.)doc");

    m.def("mul_scalar_",
          &op::mul_scalar_,
          py::arg("c"),
          py::arg("a"),
          py::arg("alpha"),
          R"doc(Out-of-place tensor-scalar multiplication into c.)doc");
}

} // namespace infinicore::ops
