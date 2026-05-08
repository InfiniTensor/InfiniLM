#pragma once

#include <pybind11/pybind11.h>

#include "infinicore.hpp"

namespace py = pybind11;

namespace infinicore::dtype {

inline void bind(py::module &m) {
    py::enum_<DataType>(m, "DataType")
        .value("BYTE", DataType::BYTE)
        .value("BOOL", DataType::BOOL)
        .value("I8", DataType::I8)
        .value("I16", DataType::I16)
        .value("I32", DataType::I32)
        .value("I64", DataType::I64)
        .value("U8", DataType::U8)
        .value("U16", DataType::U16)
        .value("U32", DataType::U32)
        .value("U64", DataType::U64)
        .value("F8", DataType::F8)
        .value("F16", DataType::F16)
        .value("F32", DataType::F32)
        .value("F64", DataType::F64)
        .value("C16", DataType::C16)
        .value("C32", DataType::C32)
        .value("C64", DataType::C64)
        .value("C128", DataType::C128)
        .value("BF16", DataType::BF16);
}

} // namespace infinicore::dtype
