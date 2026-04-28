#ifndef __QUICKGELU_CUDA_H__
#define __QUICKGELU_CUDA_H__

#include "../../../elementwise/nvidia/elementwise_nvidia.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace op::quickgelu::cuda {

typedef struct QuickGeluOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        // quickgelu(x) = x * sigmoid(1.702 * x)

        constexpr float alpha = 1.702f;

        if constexpr (std::is_same_v<T, half2>) {
            half2 ax = __hmul2(make_half2(alpha, alpha), x);
            half2 denominator = __hadd2(make_half2(1, 1), h2exp(__hneg2(ax)));
            half2 sigmoid = h2rcp(denominator);
            return __hmul2(x, sigmoid);

        } else if constexpr (std::is_same_v<T, half>) {
            half ax = __hmul(__float2half(alpha), x);
            half denominator = __hadd(__float2half(1.0f), hexp(__hneg(ax)));
            half sigmoid = hrcp(denominator);
            return __hmul(x, sigmoid);

        } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            float xf = __bfloat162float(x);
            float ax = alpha * xf;
            float s = 1.0f / (1.0f + __expf(-ax));
            return __float2bfloat16(xf * s);

        } else if constexpr (std::is_same_v<T, float>) {
            float ax = alpha * x;
            float s;
            if (ax >= 0.0f) {
                float z = expf(-ax);
                s = 1.0f / (1.0f + z);
            } else {
                float z = expf(ax);
                s = z / (1.0f + z);
            }
            return x * s;

        } else { // double
            double ax = static_cast<double>(alpha) * x;
            return x / (1.0 + exp(-ax));
        }
    }

} QuickGeluOp;

} // namespace op::quickgelu::cuda

#endif // __QUICKGELU_CUDA_H__
