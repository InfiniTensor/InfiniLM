#ifndef __TAN_CUDA_H__
#define __TAN_CUDA_H__

namespace op::tan::cuda {

typedef struct TanOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            // BF16
            const float x_f = __bfloat162float(x);
            return __float2bfloat16(__tanf(x_f));
        } else if constexpr (std::is_same_v<T, half>) {
            // FP16
            const float x_f = __half2float(x);
            return __float2half(__tanf(x_f));
        } else if constexpr (std::is_same_v<T, float>) {
            // FP32
            return __tanf(x);
        } else {
            return __tanf(x);
        }
    }
} TanOp;

} // namespace op::tan::cuda

#endif // __TAN_CUDA_H__
