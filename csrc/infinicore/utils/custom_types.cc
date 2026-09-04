#include "custom_types.h"
#include <cstdint>
#include <cstring>

float _f16_to_f32(fp16_t val) {
    uint16_t h = val._v;
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

fp16_t _f32_to_f16(float val) {
    uint32_t f32;
    memcpy(&f32, &val, sizeof(f32));               // Read the bits of the float32
    uint16_t sign = (f32 >> 16) & 0x8000;          // Extract the sign bit
    int32_t exponent = ((f32 >> 23) & 0xFF) - 127; // Extract and de-bias the exponent
    uint32_t mantissa = f32 & 0x7FFFFF;            // Extract the mantissa (fraction part)

    if (exponent >= 16) { // Special cases for Inf and NaN
        // NaN
        if (exponent == 128 && mantissa != 0) {
            return fp16_t{static_cast<uint16_t>(sign | 0x7E00)};
        }
        // Infinity
        return fp16_t{static_cast<uint16_t>(sign | 0x7C00)};
    } else if (exponent >= -14) { // Normalized case
        return fp16_t{(uint16_t)(sign | ((exponent + 15) << 10) | (mantissa >> 13))};
    } else if (exponent >= -24) {
        mantissa |= 0x800000; // Add implicit leading 1
        mantissa >>= (-14 - exponent);
        return fp16_t{(uint16_t)(sign | (mantissa >> 13))};
    } else {
        // Too small for subnormal: return signed zero
        return fp16_t{(uint16_t)sign};
    }
}

float _bf16_to_f32(bf16_t val) {
    // 只需把 bf16 放到 float32 高 16 bit，其余 16 位置 0。
    uint32_t bits32 = static_cast<uint32_t>(val._v) << 16;

    float out;
    std::memcpy(&out, &bits32, sizeof(out));
    return out;
}

bf16_t _f32_to_bf16(float val) {
    uint32_t bits32;
    std::memcpy(&bits32, &val, sizeof(bits32));

    // 截断前先加 0x7FFF，再根据第 16 位（有效位的最低位）的奇偶做 round-to-nearest-even
    const uint32_t rounding_bias = 0x00007FFF +          // 0111 1111 1111 1111
                                   ((bits32 >> 16) & 1); // 尾数的有效位的最低位奇数时 +1，即实现舍入偶数

    uint16_t bf16_bits = static_cast<uint16_t>((bits32 + rounding_bias) >> 16);

    return bf16_t{bf16_bits};
}
