#ifndef __HARDSWISH_MOORE_KERNEL_H__
#define __HARDSWISH_MOORE_KERNEL_H__

#include <cmath>
#include <type_traits>

namespace op::hardswish::moore {

typedef struct HardSwishOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half>) {
            float x_f = __half2float(x);
            float val = fminf(fmaxf(x_f + 3.0f, 0.0f), 6.0f);
            return __float2half(x_f * val * 0.16666667f);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            float x_f = __bfloat162float(x);
            float val = fminf(fmaxf(x_f + 3.0f, 0.0f), 6.0f);
            return __float2bfloat16_rn(x_f * val * 0.16666667f);
        } else if constexpr (std::is_same_v<T, float>) {
            float val = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
            return x * val * 0.16666667f;
        } else if constexpr (std::is_same_v<T, double>) {
            double val = fmin(fmax(x + 3.0, 0.0), 6.0);
            return x * val * (1.0 / 6.0);
        } else {
            float x_f = static_cast<float>(x);
            float val = fminf(fmaxf(x_f + 3.0f, 0.0f), 6.0f);
            return static_cast<T>(x_f * val * 0.16666667f);
        }
    }
} HardSwishOp;

} // namespace op::hardswish::moore

#endif // __HARDSWISH_MOORE_KERNEL_H__
