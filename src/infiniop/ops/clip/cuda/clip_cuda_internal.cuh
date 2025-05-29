#ifndef __CLIP_CUDA_H__
#define __CLIP_CUDA_H__

#include "../../../elementwise/cuda/elementwise_cuda.cuh"
#include <cuda_fp16.h>

namespace op::clip::cuda {

typedef struct ClipOp {
public:
    static constexpr size_t num_inputs = 3;

    template <typename T>
    __device__ __forceinline__ T operator()(const T &x, const T &min_val, const T &max_val) const {
        if constexpr (std::is_same_v<T, half2>) {
            return __hmax2(__hmin2(x, max_val), min_val);
        } else if constexpr (std::is_same_v<T, half>) {
            return __hmax(__hmin(x, max_val), min_val);
        } else if constexpr (std::is_same_v<T, float>) {
            return fmaxf(fminf(x, max_val), min_val);
        } else if constexpr (std::is_same_v<T, double>) {
            return fmax(fmin(x, max_val), min_val);
        } else {
            return std::max(std::min(x, max_val), min_val);
        }
    }
} ClipOp;
} // namespace op::clip::cuda

#endif // __CLIP_CUDA_H__
