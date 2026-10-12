#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "execution/trace/op_trace.hpp"

namespace py = pybind11;

namespace infinilm::execution::trace::pybind {

inline void bind(py::module &m) {
    auto analyzer_mod = m.def_submodule("analyzer",
                                        "Operator Trace Diagnostics");

    // --- Enums ---
    py::enum_<OpType>(analyzer_mod, "OpType")
        .value("UNKNOWN", OpType::UNKNOWN)
        .value("ATTENTION", OpType::ATTENTION)
        .value("FLASH_ATTENTION", OpType::FLASH_ATTENTION)
        .value("GEMM", OpType::GEMM)
        .value("LINEAR", OpType::LINEAR)
        .value("MATMUL", OpType::MATMUL)
        .value("SILU", OpType::SILU)
        .value("GELU", OpType::GELU)
        .value("RMS_NORM", OpType::RMS_NORM)
        .value("KV_CACHING", OpType::KV_CACHING)
        .value("PAGED_CACHING", OpType::PAGED_CACHING)
        .value("EMBEDDING", OpType::EMBEDDING)
        .value("ROPE", OpType::ROPE)
        .export_values();

    analyzer_mod.def(
        "trace_op_for_test",
        [](OpType op_type,
           const std::vector<size_t> &shape,
           uint8_t dtype,
           uint8_t device_type,
           int device_id) {
            traceOp(op_type, shape.data(), shape.size(), dtype, device_type, static_cast<int8_t>(device_id));
        },
        "Inject an OpTrace entry for testing",
        py::arg("op_type"),
        py::arg("shape"),
        py::arg("dtype") = 0,
        py::arg("device_type") = 0,
        py::arg("device_id") = 0);
    analyzer_mod.def(
        "clear_trace", []() {
            getGlobalOpTrace().clear();
        },
        "Clear the global OpTrace ring");
}

} // namespace infinilm::execution::trace::pybind
