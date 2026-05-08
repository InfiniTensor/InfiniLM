#pragma once

#include <infinicore.h>
#include <string>

namespace infinicore {

enum class DataType {
    BYTE = INFINI_DTYPE_BYTE,
    BOOL = INFINI_DTYPE_BOOL,
    I8 = INFINI_DTYPE_I8,
    I16 = INFINI_DTYPE_I16,
    I32 = INFINI_DTYPE_I32,
    I64 = INFINI_DTYPE_I64,
    U8 = INFINI_DTYPE_U8,
    U16 = INFINI_DTYPE_U16,
    U32 = INFINI_DTYPE_U32,
    U64 = INFINI_DTYPE_U64,
    F8 = INFINI_DTYPE_F8,
    F16 = INFINI_DTYPE_F16,
    F32 = INFINI_DTYPE_F32,
    F64 = INFINI_DTYPE_F64,
    C16 = INFINI_DTYPE_C16,
    C32 = INFINI_DTYPE_C32,
    C64 = INFINI_DTYPE_C64,
    C128 = INFINI_DTYPE_C128,
    BF16 = INFINI_DTYPE_BF16,
};

std::string toString(const DataType &dtype);
size_t dsize(const DataType &dtype);

} // namespace infinicore
