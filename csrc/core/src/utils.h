#ifndef INFINILM_CORE_UTILS_H
#define INFINILM_CORE_UTILS_H

#include "infinicore.h"
#include "utils/custom_types.h"
#include "utils/rearrange.h"

#include <string>

inline std::string infiniDtypeToString(infiniDtype_t dtype) {
    switch (dtype) {
    case INFINI_DTYPE_INVALID:
        return "INVALID";
    case INFINI_DTYPE_BYTE:
        return "BYTE";
    case INFINI_DTYPE_BOOL:
        return "BOOL";
    case INFINI_DTYPE_I8:
        return "I8";
    case INFINI_DTYPE_I16:
        return "I16";
    case INFINI_DTYPE_I32:
        return "I32";
    case INFINI_DTYPE_I64:
        return "I64";
    case INFINI_DTYPE_U8:
        return "U8";
    case INFINI_DTYPE_U16:
        return "U16";
    case INFINI_DTYPE_U32:
        return "U32";
    case INFINI_DTYPE_U64:
        return "U64";
    case INFINI_DTYPE_F8:
        return "F8";
    case INFINI_DTYPE_F16:
        return "F16";
    case INFINI_DTYPE_F32:
        return "F32";
    case INFINI_DTYPE_F64:
        return "F64";
    case INFINI_DTYPE_C16:
        return "C16";
    case INFINI_DTYPE_C32:
        return "C32";
    case INFINI_DTYPE_C64:
        return "C64";
    case INFINI_DTYPE_C128:
        return "C128";
    case INFINI_DTYPE_BF16:
        return "BF16";
    default:
        return "INVALID";
    }
}

#define CEIL_DIV(x, y) (((x) + (y)-1) / (y))

namespace utils {
inline size_t align(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}
} // namespace utils

#endif
