#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "runtime/api.hpp"

namespace py = pybind11;

namespace infinilm::execution::graph {
inline void bind(py::module_ &m) {
    py::class_<infinilm::execution::graph::Graph,
               std::shared_ptr<infinilm::execution::graph::Graph>>(m, "Graph")
        .def(py::init<>()) // allow construction
        .def("run", &infinilm::execution::graph::Graph::run)
        .def("execution_report", [](const infinilm::execution::graph::Graph &graph) {
            const auto report = graph.execution_report();
            py::dict result;
            result["operator_count"] = report.operator_count;
            result["device_graph_segments"] = report.device_graph_segments;
            result["host_segments"] = report.host_segments;
            result["capture_attempted"] = report.capture_attempted;
            result["fallback_reason"] = report.fallback_reason;
            return result;
        });
}
} // namespace infinilm::execution::graph
