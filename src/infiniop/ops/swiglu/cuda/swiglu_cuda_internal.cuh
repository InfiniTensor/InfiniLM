#ifndef __SWIGLU_CUDA_H__
#define __SWIGLU_CUDA_H__

#include "../../../elementwise/cuda/elementwise_cuda.cuh"
#include <cuda_fp16.h>

namespace op::swiglu::cuda {
typedef struct SwiGLUOp {
private:
    template <typename T>
    __device__ __forceinline__ T sigmoid(const T &x) const {
        if constexpr (std::is_same_v<T, half2>) {
            return h2rcp(__hadd2(make_half2(1, 1), h2exp(__hneg2(x))));
        } else if constexpr (std::is_same_v<T, half>) {
            return hrcp(__hadd(half(1.f), __float2half(__expf(__half2float(__hneg(x))))));
        } else if constexpr (std::is_same_v<T, float>) {
            return __frcp_rn(__fadd_rn(1, __expf(-x)));
        } else {
            return 1 / (1 + std::exp(-x));
        }
    }

public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &up, const T &gate) const {
        if constexpr (std::is_same_v<T, half2>) {
            return __hmul2(__hmul2(gate, sigmoid(gate)), up);
        } else if constexpr (std::is_same_v<T, half>) {
            return __hmul(__hmul(gate, sigmoid(gate)), up);
        } else if constexpr (std::is_same_v<T, float>) {
            return __fmul_rn(__fmul_rn(gate, sigmoid(gate)), up);
        } else {
            return gate * sigmoid(gate) * up;
        }
    }
} SwiGLUOp;
} // namespace op::swiglu::cuda

#endif // __SWIGLU_CUDA_H__
