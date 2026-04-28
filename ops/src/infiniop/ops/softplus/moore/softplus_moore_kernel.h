#ifndef __SOFTPLUS_MOORE_KERNEL_H__
#define __SOFTPLUS_MOORE_KERNEL_H__

#include <cmath>
#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <type_traits>

namespace op::softplus::moore {

struct SoftplusOp {
    static constexpr size_t num_inputs = 1;

    template <typename T>
    __device__ __forceinline__ T operator()(const T &x, float beta, float threshold) const {

        // 1. Half (FP16)
        if constexpr (std::is_same_v<T, half>) {
            float xf = __half2float(x);
            float bx = beta * xf;
            float out = (bx > threshold) ? xf : ::log1pf(::expf(bx)) / beta;
            return __float2half(out);
        } else if constexpr (std::is_same_v<T, __mt_bfloat16>) {
            float xf = __bfloat162float(x);
            float bx = beta * xf;
            float out = (bx > threshold) ? xf : ::log1pf(::expf(bx)) / beta;
            return __float2bfloat16(out);
        }
        // 3. Half2 (FP16 Vector)
        else if constexpr (std::is_same_v<T, half2>) {
            float2 xf = __half22float2(x);
            float2 out;
            float bx_x = beta * xf.x;
            out.x = (bx_x > threshold) ? xf.x : ::log1pf(::expf(bx_x)) / beta;
            float bx_y = beta * xf.y;
            out.y = (bx_y > threshold) ? xf.y : ::log1pf(::expf(bx_y)) / beta;
            return __floats2half2_rn(out.x, out.y);
        }
        // 4. Float / Double
        else {
            using CalcType = std::conditional_t<std::is_same_v<T, double>, double, float>;
            CalcType x_val = static_cast<CalcType>(x);
            CalcType b_val = static_cast<CalcType>(beta);
            CalcType t_val = static_cast<CalcType>(threshold);
            CalcType bx = b_val * x_val;

            if (bx > t_val) {
                return static_cast<T>(x_val);
            } else {
                if constexpr (std::is_same_v<CalcType, double>) {
                    return static_cast<T>(::log1p(::exp(bx)) / b_val);
                } else {
                    return static_cast<T>(::log1pf(::expf(bx)) / b_val);
                }
            }
        }
    }
};

} // namespace op::softplus::moore

#endif // __SOFTPLUS_MOORE_KERNEL_H__
