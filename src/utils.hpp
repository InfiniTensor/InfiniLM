#ifndef INFINICORE_INFER_UTILS_H
#define INFINICORE_INFER_UTILS_H
#include <infinicore.h>

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

inline void assert_true(int expr, const char *msg, const char *file, int line) {
    if (!expr) {
        fprintf(stderr, "\033[31mAssertion failed:\033[0m %s at file %s, line %d\n", msg, file, line);
        exit(EXIT_FAILURE);
    }
}

#define ASSERT(expr) assert_true((expr), #expr " is false", __FILE__, __LINE__)
#define ASSERT_EQ(a, b) assert_true((a) == (b), #a " != " #b, __FILE__, __LINE__)
#define ASSERT_VALID_PTR(a) assert_true((a) != nullptr, #a " is nullptr", __FILE__, __LINE__)

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

#endif
