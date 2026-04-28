#ifndef __MUL_MOORE_KERNEL_H__
#define __MUL_MOORE_KERNEL_H__

/*
 * This file contains the Mul operation implementation for the MUSA backend.
 *
 * It uses the 'op::mul::cuda' namespace to maintain a consistent code structure
 * and interface with the CUDA implementation, ensuring code alignment across different
 * hardware platforms.
 */

namespace op::mul::moore {
typedef struct MulOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        if constexpr (std::is_same_v<T, half2>) {
            return __hmul2(a, b);
        } else if constexpr (std::is_same_v<T, half>) {
            return __hmul(a, b);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            // On MUSA platform, convert to float, multiply, then convert back
            float a_f = __bfloat162float(a);
            float b_f = __bfloat162float(b);
            return __float2bfloat16_rn(a_f * b_f);
        } else if constexpr (std::is_same_v<T, float>) {
            // Use __fmul_rn for moore platform compatibility
            return __fmul_rn(a, b);
        } else {
            return a * b;
        }
    }
} MulOp;
} // namespace op::mul::moore

#endif // __MUL_MOORE_KERNEL_H__
