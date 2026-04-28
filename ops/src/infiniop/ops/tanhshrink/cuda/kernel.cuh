#ifndef __TANHSHRINK_CUDA_H__
#define __TANHSHRINK_CUDA_H__

namespace op::tanhshrink::cuda {

typedef struct TanhshrinkOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            // BF16
            const float x_f = __bfloat162float(x);
            return __float2bfloat16(x_f - tanhf(x_f));
        } else if constexpr (std::is_same_v<T, half>) {
            // FP16
            const float x_f = __half2float(x);
            return __float2half(x_f - tanhf(x_f));
        } else if constexpr (std::is_same_v<T, float>) {
            // FP32
            return x - tanhf(x);
        } else {
            return x - tanhf(x);
        }
    }
} TanhshrinkOp;

} // namespace op::tanhshrink::cuda

#endif // __TANHSHRINK_CUDA_H__
