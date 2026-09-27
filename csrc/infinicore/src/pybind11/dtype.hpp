#pragma once

#include <pybind11/pybind11.h>

#include "infinicore.hpp"

namespace py = pybind11;

namespace infinicore::dtype {

inline void bind(py::module &m) {
    py::enum_<DataType>(m, "DataType")
        .value("INT8", DataType::kInt8)
        .value("INT16", DataType::kInt16)
        .value("INT32", DataType::kInt32)
        .value("INT64", DataType::kInt64)
        .value("UINT8", DataType::kUInt8)
        .value("UINT16", DataType::kUInt16)
        .value("UINT32", DataType::kUInt32)
        .value("UINT64", DataType::kUInt64)
        .value("FLOAT16", DataType::kFloat16)
        .value("BFLOAT16", DataType::kBFloat16)
        .value("FLOAT32", DataType::kFloat32)
        .value("FLOAT64", DataType::kFloat64);
}

} // namespace infinicore::dtype
