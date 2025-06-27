#ifndef __SUB_CUDA_H__
#define __SUB_CUDA_H__

#include "../../../elementwise/cuda/elementwise_cuda.cuh"
#include <cuda_fp16.h>

namespace op::sub::cuda {
typedef struct SubOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        if constexpr (std::is_same_v<T, half2>) {
            return __hsub2(a, b);
        } else if constexpr (std::is_same_v<T, half>) {
            return __hsub(a, b);
        } else if constexpr (std::is_same_v<T, float>) {
            return __fsub_rd(a, b);
        } else {
            return a - b;
        }
    }
} SubOp;
} // namespace op::sub::cuda

#endif // __SUB_CUDA_H__
