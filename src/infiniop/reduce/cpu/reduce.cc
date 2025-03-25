#include "reduce.h"

namespace op::common_cpu::reduce_op {

float sum(const fp16_t *data, size_t len, ptrdiff_t stride) {
    float result = 0;
    for (size_t i = 0; i < len; i++) {
        result += utils::cast<float>(data[i * stride]);
    }

    return result;
}

float max(const fp16_t *data, size_t len, ptrdiff_t stride) {
    float result = utils::cast<float>(data[0]);
    for (size_t i = 1; i < len; i++) {
        result = std::max(result, utils::cast<float>(data[i * stride]));
    }

    return result;
}

float sumSquared(const fp16_t *data, size_t len, ptrdiff_t stride) {
    float result = 0;
    for (size_t i = 0; i < len; i++) {
        float val = utils::cast<float>(data[i * stride]);
        result += val * val;
    }

    return result;
}

} // namespace op::common_cpu::reduce_op
