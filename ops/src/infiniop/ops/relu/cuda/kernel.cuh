#ifndef __RELU_CUDA_H__
#define __RELU_CUDA_H__

#include <cmath>

namespace op::relu::cuda {

typedef struct ReluOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {

        if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            float x_f = __bfloat162float(x);
            float result = (x_f > 0.0f ? x_f : 0.0f);

            return __float2bfloat16(result);
        } else if constexpr (std::is_same_v<T, half>) {
            float x_f = __half2float(x);
            float result = (x_f > 0.0f ? x_f : 0.0f);

            return __float2half(result);
        } else if constexpr (std::is_same_v<T, float>) {

            return (x > 0.0f ? x : 0.0f);
        } else {
            return (x > 0.0 ? x : 0.0);
        }
    }
} ReluOp;

} // namespace op::relu::cuda

#endif // __RELU_CUDA_H__
