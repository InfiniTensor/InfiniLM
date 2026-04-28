#ifndef __KTHVALUE_MOORE_H__
#define __KTHVALUE_MOORE_H__

#include <cmath>
#include <cstdint>
#include <limits>
#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>

namespace op::kthvalue::moore {

template <typename T>
struct alignas(sizeof(int64_t) * 2) KeyValuePair {
    T val;
    int64_t idx;

    __device__ __forceinline__ KeyValuePair() {}
    __device__ __forceinline__ KeyValuePair(T v, int64_t i) : val(v), idx(i) {}

    __device__ __forceinline__ static KeyValuePair<T> max_value() {
        if constexpr (std::is_floating_point_v<T>) {
            return {static_cast<T>(INFINITY), -1};
        } else {
            return {static_cast<T>(1e30), -1};
        }
    }
};

template <typename T>
__device__ __forceinline__ bool is_smaller(const T &a, const T &b) {
    return a < b;
}

template <typename T>
__device__ __forceinline__ void compare_and_swap(KeyValuePair<T> &a, KeyValuePair<T> &b, bool dir) {
    bool smaller = is_smaller(a.val, b.val) || (a.val == b.val && a.idx < b.idx);

    if (smaller != dir) {
        KeyValuePair<T> tmp = a;
        a = b;
        b = tmp;
    }
}

template <typename T>
__global__ void kthvalue_kernel(
    T *__restrict__ out_values,
    int64_t *__restrict__ out_indices,
    const T *__restrict__ input,
    size_t dim_size,
    size_t inner_size,
    int k,
    size_t power_of_2_dim) {
    extern __shared__ char smem[];
    auto s_data = reinterpret_cast<KeyValuePair<T> *>(smem);

    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;

    size_t outer_idx = bid / inner_size;
    size_t inner_idx = bid % inner_size;

    size_t input_base = outer_idx * dim_size * inner_size + inner_idx;
    size_t stride = inner_size;

    for (unsigned int i = tid; i < power_of_2_dim; i += blockDim.x) {
        if (i < dim_size) {
            T val = input[input_base + i * stride];
            s_data[i] = KeyValuePair<T>(val, static_cast<int64_t>(i));
        } else {
            s_data[i] = KeyValuePair<T>::max_value();
        }
    }
    __syncthreads();

    for (unsigned int size = 2; size <= power_of_2_dim; size <<= 1) {
        bool dir = (tid & (size / 2)) == 0;

        for (unsigned int stride_step = size >> 1; stride_step > 0; stride_step >>= 1) {
            unsigned int pos = 2 * tid - (tid & (stride_step - 1));

            if (pos + stride_step < power_of_2_dim) {
                unsigned int next_pos = pos + stride_step;
                bool direction = ((pos & size) == 0);
                compare_and_swap(s_data[pos], s_data[next_pos], direction);
            }
            __syncthreads();
        }
    }

    if (tid == 0) {
        int target_k = k - 1;
        if (target_k >= 0 && target_k < dim_size) {
            out_values[bid] = s_data[target_k].val;
            out_indices[bid] = s_data[target_k].idx;
        }
    }
}

} // namespace op::kthvalue::moore

#endif // __KTHVALUE_MOORE_H__
