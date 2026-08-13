#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "infinicore.hpp"

namespace py = pybind11;

namespace infinicore::graph {
inline void bind(py::module_ &m) {
    py::class_<infinicore::graph::Graph,
               std::shared_ptr<infinicore::graph::Graph>>(m, "Graph")
        .def(py::init<>()) // allow construction
        .def("run", &infinicore::graph::Graph::run);
}
} // namespace infinicore::graph
