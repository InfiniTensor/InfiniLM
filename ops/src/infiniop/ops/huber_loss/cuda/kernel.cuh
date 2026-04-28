#ifndef __HUBER_LOSS_CUDA_CUH__
#define __HUBER_LOSS_CUDA_CUH__

#include <cmath>
#include <cstdio>

namespace op::huber_loss::cuda {

__device__ __forceinline__ float warpReduceSum(float val) {
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

__device__ __forceinline__ float blockReduceSum(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;
    if (wid == 0) {
        val = warpReduceSum(val);
    }
    return val;
}

struct HuberLossFunctor {
    float delta;
    float half_delta;

    __host__ __device__ HuberLossFunctor(float delta_val)
        : delta(delta_val), half_delta(0.5f * delta_val) {}

    __device__ __forceinline__ float compute(float input_val, float target_val) const {
        float diff = input_val - target_val;
        float abs_diff = std::abs(diff);

        if (abs_diff < delta) {
            return 0.5f * diff * diff;
        } else {
            return delta * (abs_diff - half_delta);
        }
    }
};

template <typename T>
__global__ void huber_loss_kernel(
    T *__restrict__ output,
    const T *__restrict__ input,
    const T *__restrict__ target,
    size_t count,
    HuberLossFunctor functor) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < count) {
        float in_val = static_cast<float>(input[idx]);
        float tg_val = static_cast<float>(target[idx]);

        float loss = functor.compute(in_val, tg_val);

        output[idx] = static_cast<T>(loss);
    }
}

template <typename T>
__global__ void huber_loss_reduce_kernel(
    float *output,
    const T *__restrict__ input,
    const T *__restrict__ target,
    size_t count,
    HuberLossFunctor functor,
    float scale) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    float local_sum = 0.0f;

    for (size_t i = idx; i < count; i += stride) {
        float in_val = static_cast<float>(input[i]);
        float tg_val = static_cast<float>(target[i]);

        local_sum += functor.compute(in_val, tg_val);
    }

    float block_sum = blockReduceSum(local_sum);

    if (threadIdx.x == 0) {
        atomicAdd(output, block_sum * scale);
    }
}

template <typename T>
__global__ void cast_float_to_t(T *output, const float *src) {
    *output = static_cast<T>(*src);
}

} // namespace op::huber_loss::cuda

#endif // __HUBER_LOSS_CUDA_CUH__
