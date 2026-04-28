#ifndef __HARDTANH_MOORE_KERNEL_H__
#define __HARDTANH_MOORE_KERNEL_H__

#include <cmath>
#include <type_traits>

namespace op::hardtanh::moore {

typedef struct HardTanhOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    __device__ __forceinline__ T operator()(const T &x, float min_val, float max_val) const {
        if constexpr (std::is_same_v<T, half>) {
            float x_f = __half2float(x);
            return __float2half(fminf(max_val, fmaxf(min_val, x_f)));
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            float x_f = __bfloat162float(x);
            return __float2bfloat16_rn(fminf(max_val, fmaxf(min_val, x_f)));
        } else if constexpr (std::is_same_v<T, float>) {
            return fminf(max_val, fmaxf(min_val, x));
        } else if constexpr (std::is_same_v<T, double>) {
            return fmin((double)max_val, fmax((double)min_val, x));
        } else {
            float x_f = static_cast<float>(x);
            return static_cast<T>(fminf(max_val, fmaxf(min_val, x_f)));
        }
    }
} HardTanhOp;

} // namespace op::hardtanh::moore

#endif // __HARDTANH_MOORE_KERNEL_H__
