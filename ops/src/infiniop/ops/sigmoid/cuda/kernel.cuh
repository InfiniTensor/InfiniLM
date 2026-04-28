#ifndef __SIDMOID_CUDA_H__
#define __SIDMOID_CUDA_H__

namespace op::sigmoid::cuda {
typedef struct SigmoidOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        // sigmoid(x) = 1 / (1 + exp(-x))
        if constexpr (std::is_same_v<T, half2>) {
            half2 denominator = __hadd2(make_half2(1, 1), h2exp(__hneg2(x)));
            return h2rcp(denominator);
        } else if constexpr (std::is_same_v<T, half>) {
            half denominator = __hadd(__float2half(1.0f), hexp(__hneg(x)));
            return hrcp(denominator);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            cuda_bfloat16 denominator = __float2bfloat16(__fadd_rn(1.0f, __expf(__bfloat162float(-x))));
            return __float2bfloat16(1.0f) / denominator;
        } else if constexpr (std::is_same_v<T, float>) {
            if (x >= 0.0f) {
                float z = expf(-x);
                return 1.0f / (1.0f + z);
            } else {
                float z = expf(x);
                return z / (1.0f + z);
            }
        } else { // double
            return 1.0 / (1.0 + exp(-x));
        }
    }
} SigmoidOp;
} // namespace op::sigmoid::cuda

#endif // __SIDMOID_CUDA_H__
