#include "./common_cpu.h"

float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16; // Extract the sign bit
    int32_t exponent = (h >> 10) & 0x1F;// Extract the exponent
    uint32_t mantissa = h & 0x3FF;      // Extract the mantissa (fraction part)

    if (exponent == 31) {// Special case for Inf and NaN
        if (mantissa != 0) {
            // NaN: Set float32 NaN
            uint32_t f32 = sign | 0x7F800000 | (mantissa << 13);
            return *(float *) &f32;
        } else {
            // Infinity
            uint32_t f32 = sign | 0x7F800000;
            return *(float *) &f32;
        }
    } else if (exponent == 0) {// Subnormal float16 or zero
        if (mantissa == 0) {
            // Zero (positive or negative)
            uint32_t f32 = sign;// Just return signed zero
            return *(float *) &f32;
        } else {
            // Subnormal: Convert to normalized float32
            exponent = -14;                  // Set exponent for subnormal numbers
            while ((mantissa & 0x400) == 0) {// Normalize mantissa
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x3FF;// Clear the leading 1 bit
            uint32_t f32 = sign | ((exponent + 127) << 23) | (mantissa << 13);
            return *(float *) &f32;
        }
    } else {
        // Normalized float16
        uint32_t f32 = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
        return *(float *) &f32;
    }
}

uint16_t f32_to_f16(float val) {
    uint32_t f32 = *(uint32_t *) &val;            // Read the bits of the float32
    uint16_t sign = (f32 >> 16) & 0x8000;         // Extract the sign bit
    int32_t exponent = ((f32 >> 23) & 0xFF) - 127;// Extract and de-bias the exponent
    uint32_t mantissa = f32 & 0x7FFFFF;           // Extract the mantissa (fraction part)

    if (exponent >= 31) {// Special cases for Inf and NaN
        // NaN
        if (exponent == 128 && mantissa != 0) {
            return sign | 0x7E00;
        }
        // Infinity
        return sign | 0x7C00;
    } else if (exponent >= -14) {// Normalized case
        return (uint16_t)(sign | ((exponent + 15) << 10) | (mantissa >> 13));
    } else if (exponent >= -24) {
        mantissa |= 0x800000;// Add implicit leading 1
        mantissa >>= (-14 - exponent);
        return (uint16_t)(sign | (mantissa >> 13));
    } else {
        // Too small for subnormal: return signed zero
        return (uint16_t)sign;
    }
}

size_t indexToReducedOffset(size_t flat_index, size_t ndim, int64_t const *broadcasted_strides, int64_t const *target_strides) {
    size_t res = 0;
    for (size_t i = 0; i < ndim; ++i) {
        res += flat_index / broadcasted_strides[i] * target_strides[i];
        flat_index %= broadcasted_strides[i];
    }
    return res;
}

size_t indexToOffset(size_t flat_index, size_t ndim, size_t const *shape, int64_t const *strides) {
    size_t res = 0;
    for (size_t i = ndim; i-- >= 0;) {
        res += (flat_index % shape[i]) * strides[i];
        flat_index /= shape[i];
    }
    return res;
}

size_t getPaddedSize(size_t ndim, size_t *shape, size_t const *pads) {
    uint64_t total_size = 1;
    for (size_t i = 0; i < ndim; ++i) {
        total_size *= shape[i] + (i < 2 ? 0 : 2 * pads[i - 2]);
    }
    return total_size;
}

std::vector<size_t> getPaddedShape(size_t ndim, size_t const *shape, size_t const *pads) {
    std::vector<size_t> padded_shape(ndim);
    memcpy(padded_shape.data(), shape, ndim * sizeof(size_t));
    for (size_t i = 2; i < ndim; ++i) {
        padded_shape[i] += 2 * pads[i - 2];
    }
    return std::move(padded_shape);
}
