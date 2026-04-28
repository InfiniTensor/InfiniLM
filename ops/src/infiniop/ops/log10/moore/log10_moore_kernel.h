#ifndef __LOG10_MOORE_KERNEL_H__
#define __LOG10_MOORE_KERNEL_H__

#include <cmath>
#include <type_traits>

namespace op::log10::moore {

typedef struct Log10Op {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half2>) {
            // half2 path: convert to float, compute elementwise, pack back
            float x0 = __low2float(x);
            float x1 = __high2float(x);
            return __floats2half2_rn(log10f(x0), log10f(x1));
        } else if constexpr (std::is_same_v<T, half>) {
            // FP16 path: convert to float for accuracy on MUSA backend
            float xf = __half2float(x);
            return __float2half(log10f(xf));
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            // BF16 path: compute in FP32 then cast back
            float xf = __bfloat162float(x);
            return __float2bfloat16_rn(log10f(xf));
        } else if constexpr (std::is_same_v<T, float>) {
            return log10f(x);
        } else { // double
            return ::log10(x);
        }
    }
} Log10Op;

} // namespace op::log10::moore

#endif // __LOG10_MOORE_KERNEL_H__
