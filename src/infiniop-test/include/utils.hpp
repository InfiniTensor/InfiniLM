#ifndef __INFINIOPTEST_UTILS_HPP__
#define __INFINIOPTEST_UTILS_HPP__
#include "../../utils.h"
#include "gguf.hpp"
#include <cstring>
#include <iostream>

#define CHECK_OR(cmd, action) CHECK_API_OR(cmd, INFINI_STATUS_SUCCESS, action)

inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    int32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;

    uint32_t f32;
    if (exponent == 31) {
        if (mantissa != 0) {
            f32 = sign | 0x7F800000 | (mantissa << 13);
        } else {
            f32 = sign | 0x7F800000;
        }
    } else if (exponent == 0) {
        if (mantissa == 0) {
            f32 = sign;
        } else {
            exponent = -14;
            while ((mantissa & 0x400) == 0) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x3FF;
            f32 = sign | ((exponent + 127) << 23) | (mantissa << 13);
        }
    } else {
        f32 = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }

    float result;
    memcpy(&result, &f32, sizeof(result));
    return result;
}

inline double getVal(void *ptr, GGML_TYPE ggml_type) {
    switch (ggml_type) {
    case GGML_TYPE_F16:
        return f16_to_f32(*(uint16_t *)ptr);
    case GGML_TYPE_F32:
        return *(float *)ptr;
    case GGML_TYPE_F64:
        return *(double *)ptr;
    case GGML_TYPE_I8:
        return *(int8_t *)ptr;
    case GGML_TYPE_I16:
        return *(int16_t *)ptr;
    case GGML_TYPE_I32:
        return *(int32_t *)ptr;
    case GGML_TYPE_I64:
        return (double)(*(int64_t *)ptr);
    default:
        throw std::runtime_error("Unsupported data type");
    }
}

#endif
