#ifndef __ADD_MOORE_KERNEL_H__
#define __ADD_MOORE_KERNEL_H__

/*
 * This file contains the Add operation implementation for the MUSA backend.
 *
 * It uses the 'op::add::cuda' namespace to maintain a consistent code structure
 * and interface with the CUDA implementation, ensuring code alignment across different
 * hardware platforms.
 */

namespace op::add::moore {
typedef struct AddOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        if constexpr (std::is_same_v<T, half2>) {
            return __hadd2(a, b);
        } else if constexpr (std::is_same_v<T, half>) {
            return __hadd(a, b);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            // On MUSA platform, convert to float, add, then convert back to avoid ambiguous conversion
            // from int (returned by __hadd) to __mt_bfloat16
            float a_f = __bfloat162float(a);
            float b_f = __bfloat162float(b);
            return __float2bfloat16_rn(a_f + b_f);
        } else if constexpr (std::is_same_v<T, float>) {
            // Use __fadd_rn instead of __fadd_rd for moore platform compatibility
            return __fadd_rn(a, b);
        } else {
            return a + b;
        }
    }
} AddOp;
} // namespace op::add::moore

#endif // __ADD_MOORE_KERNEL_H__
