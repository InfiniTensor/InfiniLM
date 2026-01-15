#ifndef MINICPMV_REF_OPS_HPP
#define MINICPMV_REF_OPS_HPP

#include "../../tensor.hpp"
#include "../../utils.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace minicpmv::ref_ops {

inline float read_as_f32(const void *p, infiniDtype_t dtype) {
    switch (dtype) {
    case INFINI_DTYPE_F32:
        return *reinterpret_cast<const float *>(p);
    case INFINI_DTYPE_F16:
        return f16_to_f32(*reinterpret_cast<const uint16_t *>(p));
    case INFINI_DTYPE_BF16:
        return bf16_to_f32(*reinterpret_cast<const uint16_t *>(p));
    default:
        PANIC(unsupported_dtype);
        return 0.0f;
    }
}

inline void write_from_f32(void *p, infiniDtype_t dtype, float v) {
    switch (dtype) {
    case INFINI_DTYPE_F32:
        *reinterpret_cast<float *>(p) = v;
        return;
    case INFINI_DTYPE_F16:
        *reinterpret_cast<uint16_t *>(p) = f32_to_f16(v);
        return;
    case INFINI_DTYPE_BF16:
        *reinterpret_cast<uint16_t *>(p) = f32_to_bf16(v);
        return;
    default:
        PANIC(unsupported_dtype);
        return;
    }
}

inline void layer_norm_last_dim(std::shared_ptr<Tensor> y,
                                std::shared_ptr<Tensor> x,
                                std::shared_ptr<Tensor> gamma,
                                std::shared_ptr<Tensor> beta,
                                float eps) {
    ASSERT_EQ(y->deviceType(), INFINI_DEVICE_CPU);
    ASSERT_EQ(x->deviceType(), INFINI_DEVICE_CPU);
    ASSERT_EQ(gamma->deviceType(), INFINI_DEVICE_CPU);
    ASSERT_EQ(beta->deviceType(), INFINI_DEVICE_CPU);

    ASSERT(x->ndim() >= 1);
    ASSERT_EQ(gamma->ndim(), 1);
    ASSERT_EQ(beta->ndim(), 1);
    ASSERT_EQ(gamma->shape()[0], beta->shape()[0]);

    const size_t d = gamma->shape()[0];
    ASSERT_EQ(x->shape().back(), d);
    ASSERT_EQ(y->shape(), x->shape());

    const infiniDtype_t dt_x = x->dtype();
    const infiniDtype_t dt_y = y->dtype();
    const infiniDtype_t dt_w = gamma->dtype();
    ASSERT_EQ(beta->dtype(), dt_w);

    const size_t outer = x->numel() / d;
    const char *x_ptr = reinterpret_cast<const char *>(x->data());
    char *y_ptr = reinterpret_cast<char *>(y->data());
    const char *g_ptr = reinterpret_cast<const char *>(gamma->data());
    const char *b_ptr = reinterpret_cast<const char *>(beta->data());

    const size_t sx = dsize(dt_x);
    const size_t sy = dsize(dt_y);
    const size_t sw = dsize(dt_w);

    for (size_t row = 0; row < outer; ++row) {
        float mean = 0.0f;
        for (size_t i = 0; i < d; ++i) {
            mean += read_as_f32(x_ptr + (row * d + i) * sx, dt_x);
        }
        mean /= static_cast<float>(d);

        float var = 0.0f;
        for (size_t i = 0; i < d; ++i) {
            float v = read_as_f32(x_ptr + (row * d + i) * sx, dt_x) - mean;
            var += v * v;
        }
        var /= static_cast<float>(d);

        const float inv_std = 1.0f / std::sqrt(var + eps);
        for (size_t i = 0; i < d; ++i) {
            const float xv = read_as_f32(x_ptr + (row * d + i) * sx, dt_x);
            const float gv = read_as_f32(g_ptr + i * sw, dt_w);
            const float bv = read_as_f32(b_ptr + i * sw, dt_w);
            const float yn = (xv - mean) * inv_std;
            write_from_f32(y_ptr + (row * d + i) * sy, dt_y, yn * gv + bv);
        }
    }
}

inline void layer_norm_last_dim_raw(void *y,
                                    const void *x,
                                    const void *gamma,
                                    const void *beta,
                                    infiniDtype_t dtype,
                                    size_t outer,
                                    size_t d,
                                    float eps) {
    ASSERT_VALID_PTR(y);
    ASSERT_VALID_PTR(x);
    ASSERT_VALID_PTR(gamma);
    ASSERT_VALID_PTR(beta);

    const char *x_ptr = reinterpret_cast<const char *>(x);
    char *y_ptr = reinterpret_cast<char *>(y);
    const char *g_ptr = reinterpret_cast<const char *>(gamma);
    const char *b_ptr = reinterpret_cast<const char *>(beta);

    const size_t s = dsize(dtype);

    for (size_t row = 0; row < outer; ++row) {
        float mean = 0.0f;
        for (size_t i = 0; i < d; ++i) {
            mean += read_as_f32(x_ptr + (row * d + i) * s, dtype);
        }
        mean /= static_cast<float>(d);

        float var = 0.0f;
        for (size_t i = 0; i < d; ++i) {
            float v = read_as_f32(x_ptr + (row * d + i) * s, dtype) - mean;
            var += v * v;
        }
        var /= static_cast<float>(d);

        const float inv_std = 1.0f / std::sqrt(var + eps);
        for (size_t i = 0; i < d; ++i) {
            const float xv = read_as_f32(x_ptr + (row * d + i) * s, dtype);
            const float gv = read_as_f32(g_ptr + i * s, dtype);
            const float bv = read_as_f32(b_ptr + i * s, dtype);
            const float yn = (xv - mean) * inv_std;
            write_from_f32(y_ptr + (row * d + i) * s, dtype, yn * gv + bv);
        }
    }
}

inline void softmax_last_dim_inplace(std::shared_ptr<Tensor> x) {
    ASSERT_EQ(x->deviceType(), INFINI_DEVICE_CPU);
    ASSERT(x->ndim() >= 1);
    const size_t d = x->shape().back();
    const size_t outer = x->numel() / d;
    const infiniDtype_t dt = x->dtype();
    const size_t s = dsize(dt);
    char *ptr = reinterpret_cast<char *>(x->data());

    std::vector<float> tmp(d);
    for (size_t row = 0; row < outer; ++row) {
        float max_v = -INFINITY;
        for (size_t i = 0; i < d; ++i) {
            float v = read_as_f32(ptr + (row * d + i) * s, dt);
            tmp[i] = v;
            if (v > max_v) {
                max_v = v;
            }
        }

        float sum = 0.0f;
        for (size_t i = 0; i < d; ++i) {
            float e = std::exp(tmp[i] - max_v);
            tmp[i] = e;
            sum += e;
        }
        const float inv = 1.0f / sum;

        for (size_t i = 0; i < d; ++i) {
            write_from_f32(ptr + (row * d + i) * s, dt, tmp[i] * inv);
        }
    }
}

inline float gelu_tanh_f32(float x) {
    // gelu(x) ≈ 0.5x(1+tanh(sqrt(2/pi)(x+0.044715x^3)))
    constexpr float kAlpha = 0.7978845608028654f; // sqrt(2/pi)
    constexpr float kBeta = 0.044715f;
    const float x3 = x * x * x;
    const float u = kAlpha * (x + kBeta * x3);
    return 0.5f * x * (1.0f + std::tanh(u));
}

inline void gelu_tanh_inplace(std::shared_ptr<Tensor> x) {
    ASSERT_EQ(x->deviceType(), INFINI_DEVICE_CPU);
    const infiniDtype_t dt = x->dtype();
    const size_t n = x->numel();
    const size_t s = dsize(dt);
    char *ptr = reinterpret_cast<char *>(x->data());
    for (size_t i = 0; i < n; ++i) {
        float v = read_as_f32(ptr + i * s, dt);
        v = gelu_tanh_f32(v);
        write_from_f32(ptr + i * s, dt, v);
    }
}

} // namespace minicpmv::ref_ops

#endif
