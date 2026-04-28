#pragma once
#include <cmath>
#include <limits>
#include <type_traits>

namespace op::cuda {

constexpr float PI_F = 3.14159265358979323846f;
constexpr double PI = 3.14159265358979323846;

// Inverse error function.
__device__ __forceinline__ float erfinv_impl(float x) {
    if (x == 1.0f) {
        return std::numeric_limits<float>::infinity();
    }
    if (x == -1.0f) {
        return -std::numeric_limits<float>::infinity();
    }
    if (x > 1.0f || x < -1.0f) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    if (x == 0.0f) {
        return 0.0f;
    }

    // Winitzki approximation
    const float a = 0.147f;
    const float ln = log1pf(-x * x);
    const float t = 2.0f / (PI_F * a) + ln * 0.5f;

    float inside = t * t - ln / a;
    inside = inside > 0.0f ? inside : 0.0f;

    float y0 = copysignf(sqrtf(sqrtf(inside) - t), x);

    float y = y0;
    const float sqrt_pi_f = sqrtf(PI_F);

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float erf_y = erff(y);
        const float derf_dy = 2.0f / sqrt_pi_f * expf(-y * y);
        y = y - (erf_y - x) / derf_dy;
    }

    // Slow path near |x| ~ 1
    const float ax = fabsf(x);
    if (1.0f - ax < 1e-4f) {
        const double xd = static_cast<double>(x);
        double yd = static_cast<double>(y);
        const double sqrt_pi = sqrt(PI);

#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const double erf_y = erf(yd);
            const double derf_dy = 2.0 / sqrt_pi * exp(-yd * yd);
            yd = yd - (erf_y - xd) / derf_dy;
        }
        y = static_cast<float>(yd);
    }

    return y;
}

struct ErfinvOp {
    static constexpr size_t num_inputs = 1;

    template <typename T>
    __device__ __forceinline__ T operator()(T x) const {
        if constexpr (std::is_same_v<T, float>) {
            return erfinv_impl(x);

        } else if constexpr (std::is_same_v<T, double>) {
            if (x == 1.0) {
                return std::numeric_limits<double>::infinity();
            }
            if (x == -1.0) {
                return -std::numeric_limits<double>::infinity();
            }
            if (x > 1.0 || x < -1.0) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            if (x == 0.0) {
                return 0.0;
            }

            const double a = 0.147;
            const double ln = log1p(-x * x);
            const double t = 2.0 / (PI * a) + ln * 0.5;

            double inside = t * t - ln / a;
            inside = inside > 0.0 ? inside : 0.0;

            double y = copysign(sqrt(sqrt(inside) - t), x);

            const int max_iter = 30;
            const double tol = 1e-14;
            const double sqrt_pi = sqrt(PI);

            for (int i = 0; i < max_iter; ++i) {
                const double erf_y = erf(y);
                const double error = erf_y - x;
                if (fabs(error) < tol) {
                    break;
                }
                const double derf_dy = 2.0 / sqrt_pi * exp(-y * y);
                y = y - error / derf_dy;
            }
            return y;

        } else {
            // F16 / BF16 / other types
            float xf;
            if constexpr (std::is_same_v<T, half>) {
                xf = __half2float(x);
                return __float2half_rn(erfinv_impl(xf));
            } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
                xf = __bfloat162float(x);
                return __float2bfloat16_rn(erfinv_impl(xf));
            } else {
                xf = static_cast<float>(x);
                return static_cast<T>(erfinv_impl(xf));
            }
        }
    }
};

} // namespace op::cuda
