#pragma once

#include "infinicore/dtype.hpp"

#include <infiniccl/infiniccl.h>

#include <stdexcept>
#include <string>

namespace infinicore::op::distributed::detail {

inline infinicclDataType_t toInfinicclDataType(DataType dtype) {
    switch (dtype) {
    case DataType::kInt8:
        return infinicclInt8;
    case DataType::kInt16:
        return infinicclInt16;
    case DataType::kInt32:
        return infinicclInt32;
    case DataType::kInt64:
        return infinicclInt64;
    case DataType::kUInt8:
        return infinicclUInt8;
    case DataType::kUInt16:
        return infinicclUInt16;
    case DataType::kUInt32:
        return infinicclUInt32;
    case DataType::kUInt64:
        return infinicclUInt64;
    case DataType::kFloat16:
        return infinicclFloat16;
    case DataType::kBFloat16:
        return infinicclBFloat16;
    case DataType::kFloat32:
        return infinicclFloat32;
    case DataType::kFloat64:
        return infinicclFloat64;
    }
    throw std::invalid_argument("unsupported data type for InfiniCCL");
}

inline void checkInfiniccl(const char *operation, infinicclResult_t result) {
    if (result == infinicclSuccess) {
        return;
    }

    const auto result_string = std::to_string(static_cast<int>(result));
    if (result == infinicclNotSupported) {
        throw std::runtime_error("InfiniCCL operation `" + std::string(operation)
                                 + "` is not supported (result " + result_string + ")");
    }
    throw std::runtime_error("InfiniCCL operation `" + std::string(operation)
                             + "` failed with result " + result_string);
}

} // namespace infinicore::op::distributed::detail
