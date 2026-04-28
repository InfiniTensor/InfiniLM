#include "reduce.h"

namespace op::common_cpu::reduce_op {

template <typename HalfType>
float sum_half_impl(const HalfType *data, size_t len, ptrdiff_t stride) {
    float result = 0;
    for (size_t i = 0; i < len; i++) {
        result += utils::cast<float>(data[i * stride]);
    }
    return result;
}

template <typename HalfType>
float max_half_impl(const HalfType *data, size_t len, ptrdiff_t stride) {
    float result = utils::cast<float>(data[0]);
    for (size_t i = 1; i < len; i++) {
        result = std::max(result, utils::cast<float>(data[i * stride]));
    }
    return result;
}

template <typename HalfType>
float sumSquared_half_impl(const HalfType *data, size_t len, ptrdiff_t stride) {
    float result = 0;
    for (size_t i = 0; i < len; i++) {
        float val = utils::cast<float>(data[i * stride]);
        result += val * val;
    }
    return result;
}

// fp16
float sum(const fp16_t *data, size_t len, ptrdiff_t stride) {
    return sum_half_impl(data, len, stride);
}

float max(const fp16_t *data, size_t len, ptrdiff_t stride) {
    return max_half_impl(data, len, stride);
}

float sumSquared(const fp16_t *data, size_t len, ptrdiff_t stride) {
    return sumSquared_half_impl(data, len, stride);
}

// bf16
float sum(const bf16_t *data, size_t len, ptrdiff_t stride) {
    return sum_half_impl(data, len, stride);
}

float max(const bf16_t *data, size_t len, ptrdiff_t stride) {
    return max_half_impl(data, len, stride);
}

float sumSquared(const bf16_t *data, size_t len, ptrdiff_t stride) {
    return sumSquared_half_impl(data, len, stride);
}

} // namespace op::common_cpu::reduce_op
