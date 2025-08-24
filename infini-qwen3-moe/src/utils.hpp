#ifndef INFINICORE_INFER_UTILS_H
#define INFINICORE_INFER_UTILS_H
#include <infinicore.h>

#include <cstring>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

inline void assertTrue(int expr, const char *msg, const char *file, int line) {
    if (!expr) {
        fprintf(stderr, "\033[31mAssertion failed:\033[0m %s at file %s, line %d\n", msg, file, line);
        exit(EXIT_FAILURE);
    }
}

#define ASSERT(expr) assertTrue((expr), #expr " is false", __FILE__, __LINE__)
#define ASSERT_EQ(a, b) assertTrue((a) == (b), #a " != " #b, __FILE__, __LINE__)
#define ASSERT_VALID_PTR(a) assertTrue((a) != nullptr, #a " is nullptr", __FILE__, __LINE__)

#define PANIC(EXPR)                                             \
    printf("Error at %s:%d - %s\n", __FILE__, __LINE__, #EXPR); \
    exit(EXIT_FAILURE)

#define RUN_INFINI(API)                                                         \
    do {                                                                        \
        auto api_result_ = (API);                                               \
        if (api_result_ != INFINI_STATUS_SUCCESS) {                             \
            std::cerr << "Error Code " << api_result_ << " in `" << #API << "`" \
                      << " from " << __func__                                   \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;    \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;  // Extract the sign bit
    int32_t exponent = (h >> 10) & 0x1F; // Extract the exponent
    uint32_t mantissa = h & 0x3FF;       // Extract the mantissa (fraction part)

    if (exponent == 31) { // Special case for Inf and NaN
        if (mantissa != 0) {
            // NaN: Set float32 NaN
            uint32_t f32 = sign | 0x7F800000 | (mantissa << 13);
            return *(float *)&f32;
        } else {
            // Infinity
            uint32_t f32 = sign | 0x7F800000;
            return *(float *)&f32;
        }
    } else if (exponent == 0) { // Subnormal float16 or zero
        if (mantissa == 0) {
            // Zero (positive or negative)
            uint32_t f32 = sign; // Just return signed zero
            return *(float *)&f32;
        } else {
            // Subnormal: Convert to normalized float32
            exponent = -14;                   // Set exponent for subnormal numbers
            while ((mantissa & 0x400) == 0) { // Normalize mantissa
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x3FF; // Clear the leading 1 bit
            uint32_t f32 = sign | ((exponent + 127) << 23) | (mantissa << 13);
            return *(float *)&f32;
        }
    } else {
        // Normalized float16
        uint32_t f32 = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
        return *(float *)&f32;
    }
}

inline uint16_t f32_to_f16(float val) {
    uint32_t f32;
    memcpy(&f32, &val, sizeof(f32));               // Read the bits of the float32
    uint16_t sign = (f32 >> 16) & 0x8000;          // Extract the sign bit
    int32_t exponent = ((f32 >> 23) & 0xFF) - 127; // Extract and de-bias the exponent
    uint32_t mantissa = f32 & 0x7FFFFF;            // Extract the mantissa (fraction part)

    if (exponent >= 31) { // Special cases for Inf and NaN
        // NaN
        if (exponent == 128 && mantissa != 0) {
            return static_cast<uint16_t>(sign | 0x7E00);
        }
        // Infinity
        return static_cast<uint16_t>(sign | 0x7C00);
    } else if (exponent >= -14) { // Normalized case
        return (uint16_t)(sign | ((exponent + 15) << 10) | (mantissa >> 13));
    } else if (exponent >= -24) {
        mantissa |= 0x800000; // Add implicit leading 1
        mantissa >>= (-14 - exponent);
        return (uint16_t)(sign | (mantissa >> 13));
    } else {
        // Too small for subnormal: return signed zero
        return (uint16_t)sign;
    }
}

inline float bf16_to_f32(uint16_t val) {
    // 只需把 bf16 放到 float32 高 16 bit，其余 16 位置 0。
    uint32_t bits32 = static_cast<uint32_t>(val) << 16;

    float out;
    std::memcpy(&out, &bits32, sizeof(out));
    return out;
}

inline uint16_t f32_to_bf16(float val) {
    uint32_t bits32;
    std::memcpy(&bits32, &val, sizeof(bits32));

    // 截断前先加 0x7FFF，再根据第 16 位（有效位的最低位）的奇偶做 round-to-nearest-even
    const uint32_t rounding_bias = 0x00007FFF +          // 0111 1111 1111 1111
                                   ((bits32 >> 16) & 1); // 尾数的有效位的最低位奇数时 +1，即实现舍入偶数

    uint16_t bf16_bits = static_cast<uint16_t>((bits32 + rounding_bias) >> 16);

    return bf16_bits;
}

#endif
