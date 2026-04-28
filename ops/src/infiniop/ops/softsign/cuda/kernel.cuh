#ifndef __SOFTSIGN_CUDA_H__
#define __SOFTSIGN_CUDA_H__

#include <cmath>
#include <type_traits>

namespace op::softsign::cuda {

struct SoftsignOp {
    static constexpr size_t num_inputs = 1;

    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half2>) {
            const half2 one = __float2half2_rn(1.0f);
            const half2 abs_x = __habs2(x);
            const half2 denom = __hadd2(one, abs_x);
            return __h2div(x, denom);
        } else if constexpr (std::is_same_v<T, half>) {
#if __CUDA_ARCH__ >= 530
            const half one = __float2half(1.0f);
            return __hdiv(x, __hadd(one, __habs(x)));
#else
            return static_cast<half>(static_cast<float>(x) / (1.0f + fabsf(static_cast<float>(x))));
#endif
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            // Avoid __habs which is for fp16. Use manual abs or operators to keep bf16 precision.
            const T abs_x = (x >= T(0.0f)) ? x : -x;
            return x / (T(1.0f) + abs_x);
        } else if constexpr (std::is_same_v<T, float>) {
            return x / (1.0f + fabsf(x));
        } else {
            return x / (static_cast<T>(1) + std::abs(x));
        }
    }
};

} // namespace op::softsign::cuda

#endif // __SOFTSIGN_CUDA_H__
