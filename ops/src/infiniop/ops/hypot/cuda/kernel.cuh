#ifndef __HYPOT_CUDA_H__
#define __HYPOT_CUDA_H__

#include <cmath>
#include <type_traits>

namespace op::hypot::cuda {

typedef struct HypotOp {
public:
    static constexpr size_t num_inputs = 2;

    template <typename T>
    __device__ __forceinline__ T operator()(const T &x, const T &y) const {

        if constexpr (std::is_same_v<T, float>) {
            return sqrtf(fmaf(x, x, y * y));
        } else if constexpr (std::is_same_v<T, half2>) {
            half2 sq_sum = __hfma2(x, x, __hmul2(y, y));
            return h2sqrt(sq_sum);
        }

        else if constexpr (std::is_same_v<T, half>) {
            return hsqrt(__hfma(x, x, __hmul(y, y)));
        } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {

            float f0_x = __bfloat162float(__low2bfloat16(x));
            float f1_x = __bfloat162float(__high2bfloat16(x));

            float f0_y = __bfloat162float(__low2bfloat16(y));
            float f1_y = __bfloat162float(__high2bfloat16(y));
            float res0 = sqrtf(fmaf(f0_x, f0_x, f0_y * f0_y));
            float res1 = sqrtf(fmaf(f1_x, f1_x, f1_y * f1_y));

            return __floats2bfloat162_rn(res0, res1);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            float fx = __bfloat162float(x);
            float fy = __bfloat162float(y);
            return __float2bfloat16(sqrtf(fmaf(fx, fx, fy * fy)));
        }

        else if constexpr (std::is_same_v<T, double>) {
            return sqrt(fma(x, x, y * y));
        } else {
            return static_cast<T>(sqrt(fma(static_cast<double>(x), static_cast<double>(x), static_cast<double>(y) * static_cast<double>(y))));
        }
    }
} HypotOp;

} // namespace op::hypot::cuda

#endif // __HYPOT_CUDA_H__
