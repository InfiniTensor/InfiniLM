#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/gaussian_nll_loss.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_gaussian_nll_loss(py::module &m) {
    m.def("gaussian_nll_loss",
          &op::gaussian_nll_loss,
          py::arg("input"),
          py::arg("target"),
          py::arg("var"),
          py::arg("full") = false,
          py::arg("eps") = 1e-6,
          py::arg("reduction") = 1,
          R"doc(Gaussian negative log-likelihood loss.)doc");

    m.def("gaussian_nll_loss_",
          &op::gaussian_nll_loss_,
          py::arg("out"),
          py::arg("input"),
          py::arg("target"),
          py::arg("var"),
          py::arg("full") = false,
          py::arg("eps") = 1e-6,
          py::arg("reduction") = 1,
          R"doc(In-place Gaussian negative log-likelihood loss.)doc");
}

} // namespace infinicore::ops
