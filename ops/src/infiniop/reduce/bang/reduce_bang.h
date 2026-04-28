#ifndef __INFINIOP_REDUCE_BANG_H__
#define __INFINIOP_REDUCE_BANG_H__

#include "../../devices/bang/common_bang.h"

namespace op::common_bang::reduce_op {

constexpr int batch_size = 128 / sizeof(float);

__mlu_func__ void sumInternal(float *dst, float *src, int max_batch) {
    const int width = max_batch / batch_size;

    // Use vectorized reduction
    if (width >= 4) {
        __bang_sumpool(
            dst, src,
            batch_size, 1, width,
            1, width, 1, 1);
        __bang_reduce_sum(dst, dst, batch_size);
    } else {
        // Fallback for small batches
        float sum = 0.0f;
        for (int i = 0; i < max_batch; ++i) {
            sum += src[i];
        }
        dst[0] = sum;
    }
}

template <typename T>
__mlu_func__ void sumTyped(float *result, T *data, size_t len) {
    if constexpr (std::is_same_v<T, half>) {
        __bang_half2float((float *)data, data + len, len);
        sumInternal(result, (float *)data, len);
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        __bang_bfloat162float((float *)data, data + len, len);
        sumInternal(result, (float *)data, len);
    } else {
        sumInternal(result, data, len);
    }
}

template <typename T>
__mlu_func__ float sum(const T *source, T *src, float *dst, int num_elements, int max_batch) {
    float res = 0.0f;
    int offset = (sizeof(T) == 2 ? max_batch : 0);

    size_t processed = 0;
    while (processed < num_elements) {
        size_t curr_batch = std::min<size_t>(max_batch, num_elements - processed);

        if (curr_batch < max_batch) {
            __bang_write_value(src, max_batch + offset, 0);
        }

        __memcpy(src + offset, source + processed, curr_batch * sizeof(T), GDRAM2NRAM);
        sumTyped(dst, src, max_batch);
        res += dst[0];
        processed += curr_batch;
    }

    return res;
}

template <typename T>
__mlu_func__ float sumBatched(const T *source, T *src, float *dst, int num_elements, int max_batch) {
    constexpr int min_vector_size = 32;

    // For small vectors, use safer element-wise computation
    if (num_elements < min_vector_size) {
        return sum(source, src, dst, num_elements, max_batch);
    }

    float res = 0.0f;
    int offset = (sizeof(T) == 2 ? max_batch : 0);

    size_t processed = 0;
    while (processed < num_elements) {
        size_t curr_batch = std::min<size_t>(max_batch, num_elements - processed);
        size_t aligned_batch = (curr_batch / batch_size) * batch_size;
        size_t remainder = curr_batch % batch_size;

        // Ensure NRAM buffer is zeroed
        __bang_write_value(src, max_batch + offset, 0);

        // Copy data to NRAM
        __memcpy(src + offset, source + processed, curr_batch * sizeof(T), GDRAM2NRAM);

        if constexpr (std::is_same_v<T, half>) {
            __bang_half2float((float *)(src + offset), src + offset, curr_batch);
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            __bang_bfloat162float((float *)(src + offset), src + offset, curr_batch);
        }

        // Process aligned portion
        if (aligned_batch > 0) {
            sumInternal(dst, (float *)(src + offset), aligned_batch);
            res += dst[0];
        }
        // Process unaligned tail
        if (remainder > 0) {
            for (size_t i = aligned_batch; i < curr_batch; ++i) {
                res += ((float *)(src + offset))[i];
            }
        }

        processed += curr_batch;
    }

    return res;
}

template <typename T>
__mlu_func__ float sumSquared(const T *source, T *src, float *dst, int num_elements, int max_batch) {
    float res = 0.0f;
    int offset = (sizeof(T) == 2 ? max_batch : 0);

    size_t processed = 0;
    while (processed < num_elements) {
        size_t curr_batch = std::min<size_t>(max_batch, num_elements - processed);

        if (curr_batch < max_batch) {
            __bang_write_value(src, max_batch + offset, 0);
        }

        __memcpy(src + offset, source + processed, curr_batch * sizeof(T), GDRAM2NRAM);

        float sum = 0.0f;

        for (size_t i = 0; i < curr_batch; ++i) {
            float val = 0.0f;
            if constexpr (std::is_same_v<T, half>) {
                val = __half2float(src[offset + i]);
            } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                val = __bfloat162float(src[offset + i]);
            } else {
                val = src[offset + i];
            }
            sum += val * val;
        }

        res += sum;
        processed += curr_batch;
    }

    return res;
}

template <typename T>
__mlu_func__ float sumSquaredBatched(const T *source, T *src, float *dst, int num_elements, int max_batch) {
    constexpr int min_vector_size = 32;

    // For small vectors, use safer element-wise computation
    if (num_elements < min_vector_size) {
        return sumSquared(source, src, dst, num_elements, max_batch);
    }

    float res = 0.0f;
    int offset = (sizeof(T) == 2 ? max_batch : 0);

    size_t processed = 0;
    while (processed < num_elements) {
        size_t curr_batch = std::min<size_t>(max_batch, num_elements - processed);
        size_t aligned_batch = (curr_batch / batch_size) * batch_size;
        size_t remainder = curr_batch % batch_size;

        // Ensure NRAM buffer is zeroed
        __bang_write_value(src, max_batch + offset, 0);

        // Copy data to NRAM
        __memcpy(src + offset, source + processed, curr_batch * sizeof(T), GDRAM2NRAM);

        if constexpr (std::is_same_v<T, half>) {
            __bang_half2float((float *)(src + offset), src + offset, curr_batch);
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            __bang_bfloat162float((float *)(src + offset), src + offset, curr_batch);
        }

        // Process aligned portion
        if (aligned_batch > 0) {
            __bang_mul((float *)(src + offset), (float *)(src + offset),
                       (float *)(src + offset), aligned_batch);
            sumInternal(dst, (float *)(src + offset), aligned_batch);
            res += dst[0];
        }

        // Process unaligned tail
        if (remainder > 0) {
            for (size_t i = aligned_batch; i < curr_batch; ++i) {
                float val = ((float *)(src + offset))[i];
                res += val * val;
            }
        }

        processed += curr_batch;
    }

    return res;
}

__mlu_func__ void maxInternal(float *dst, float *src, int max_batch) {
    __bang_maxpool(
        dst, src,
        batch_size,             // channel size
        1,                      // height
        max_batch / batch_size, // width
        1,                      // kernel_height
        max_batch / batch_size, // kernel_width
        1,                      // stride_height
        1                       // stride_width
    );
    __bang_argmax(dst, dst, batch_size);
}

template <typename T>
__mlu_func__ void maxTyped(float *result, T *data, size_t len) {
    if constexpr (std::is_same_v<T, half>) {
        __bang_half2float((float *)data, data + len, len);
        maxInternal(result, (float *)data, len);
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        __bang_bfloat162float((float *)data, data + len, len);
        maxInternal(result, (float *)data, len);
    } else {
        maxInternal(result, data, len);
    }
}

template <typename T>
__mlu_func__ float max(const T *source, T *src, float *dst, int num_elements, int max_batch) {
    float max_val = -INFINITY;
    int offset = (sizeof(T) == 2 ? max_batch : 0);

    size_t processed = 0;
    while (processed < num_elements) {
        size_t curr_batch = std::min<size_t>(max_batch, num_elements - processed);

        if (curr_batch < max_batch) {
            __bang_write_value(src, max_batch + offset, 0);
        }

        __memcpy(src + offset, source + processed, curr_batch * sizeof(T), GDRAM2NRAM);
        maxTyped(dst, src, max_batch);
        max_val = std::max(max_val, dst[0]);
        processed += curr_batch;
    }

    return max_val;
}

template <typename T>
__mlu_func__ float maxBatched(const T *source, T *src, float *dst, int num_elements, int max_batch) {
    constexpr int min_vector_size = 32;

    // For small vectors, use safer element-wise computation
    if (num_elements < min_vector_size) {
        return max(source, src, dst, num_elements, max_batch);
    }

    float max_val = -INFINITY;
    int offset = (sizeof(T) == 2 ? max_batch : 0);

    size_t processed = 0;
    while (processed < num_elements) {
        size_t curr_batch = std::min<size_t>(max_batch, num_elements - processed);

        if (curr_batch < max_batch) {
            __bang_write_value(src, max_batch + offset, 0);
        }

        __memcpy(src + offset, source + processed, curr_batch * sizeof(T), GDRAM2NRAM);
        maxTyped(dst, src, max_batch);
        max_val = std::max(max_val, dst[0]);
        processed += curr_batch;
    }

    return max_val;
}

} // namespace op::common_bang::reduce_op

#endif // __INFINIOP_REDUCE_BANG_H__
