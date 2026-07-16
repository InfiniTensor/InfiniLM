#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "infinicore/ops/mrope.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_mrope(py::module &m) {
    m.def("mrope",
          &op::mrope,
          py::arg("q"),
          py::arg("k"),
          py::arg("cos"),
          py::arg("sin"),
          py::arg("positions"),
          py::arg("head_size"),
          py::arg("rotary_dim"),
          py::arg("section_t"),
          py::arg("section_h"),
          py::arg("section_w"),
          py::arg("interleaved"),
          R"doc(Multimodal rotary position embedding for q and k.)doc");

    m.def("mrope_",
          &op::mrope_,
          py::arg("q_out"),
          py::arg("k_out"),
          py::arg("q"),
          py::arg("k"),
          py::arg("cos"),
          py::arg("sin"),
          py::arg("positions"),
          py::arg("head_size"),
          py::arg("rotary_dim"),
          py::arg("section_t"),
          py::arg("section_h"),
          py::arg("section_w"),
          py::arg("interleaved"),
          R"doc(In-place multimodal rotary position embedding for q and k.)doc");
}

} // namespace infinicore::ops
