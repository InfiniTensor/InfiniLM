#include <infinicore.hpp>

namespace infinicore {

std::string toString(const DataType &dtype) {
    switch (dtype) {
    case DataType::BYTE:
        return "BYTE";
    case DataType::BOOL:
        return "BOOL";
    case DataType::I8:
        return "I8";
    case DataType::I16:
        return "I16";
    case DataType::I32:
        return "I32";
    case DataType::I64:
        return "I64";
    case DataType::U8:
        return "U8";
    case DataType::U16:
        return "U16";
    case DataType::U32:
        return "U32";
    case DataType::U64:
        return "U64";
    case DataType::F8:
        return "F8";
    case DataType::F16:
        return "F16";
    case DataType::F32:
        return "F32";
    case DataType::F64:
        return "F64";
    case DataType::C16:
        return "C16";
    case DataType::C32:
        return "C32";
    case DataType::C64:
        return "C64";
    case DataType::C128:
        return "C128";
    case DataType::BF16:
        return "BF16";
    }

    // TODO: Add error handling.
    return "";
}

size_t dsize(const DataType &dtype) {
    switch (dtype) {
    case DataType::BYTE:
    case DataType::BOOL:
    case DataType::F8:
    case DataType::I8:
    case DataType::U8:
        return 1;
    case DataType::I16:
    case DataType::U16:
    case DataType::F16:
    case DataType::BF16:
    case DataType::C16:
        return 2;
    case DataType::I32:
    case DataType::U32:
    case DataType::F32:
    case DataType::C32:
        return 4;
    case DataType::I64:
    case DataType::U64:
    case DataType::F64:
    case DataType::C64:
        return 8;
    case DataType::C128:
        return 16;
    }

    // TODO: Add error handling.
    return 0;
}

} // namespace infinicore
