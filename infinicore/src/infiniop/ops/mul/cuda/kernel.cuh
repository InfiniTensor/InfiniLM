#ifndef __MUL_CUDA_H__
#define __MUL_CUDA_H__

namespace op::mul::cuda {
typedef struct MulOp {
    static constexpr size_t num_inputs = 2;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        if constexpr (std::is_same_v<T, half2> || std::is_same_v<T, cuda_bfloat162>) {
            return __hmul2(a, b);
        } else if constexpr (std::is_same_v<T, half> || std::is_same_v<T, cuda_bfloat16>) {
            return __hmul(a, b);
        } else if constexpr (std::is_same_v<T, float>) {
            return __fmul_rn(a, b);
        } else {
            return a * b;
        }
    }
} MulOp;

} // namespace op::mul::cuda

#endif // __MUL_CUDA_H__
