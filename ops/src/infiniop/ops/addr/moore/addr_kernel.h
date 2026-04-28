#ifndef __ADDR_MOORE_H__
#define __ADDR_MOORE_H__

template <typename T>
__device__ void addr_kernel(T *out, const T *input, const T *vec1,
                            const T *vec2, size_t n, size_t m, float beta,
                            float alpha, ptrdiff_t stride1, ptrdiff_t stride2,
                            ptrdiff_t out_stride_0, ptrdiff_t out_stride_1,
                            ptrdiff_t in_stride_0, ptrdiff_t in_stride_1) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= m) {
        return;
    }
    size_t out_idx = i * out_stride_0 + j * out_stride_1;
    size_t in_idx = i * in_stride_0 + j * in_stride_1;
    size_t vec1_idx = i * stride1;
    size_t vec2_idx = j * stride2;
    T in_val = input[in_idx];
    T vec1_val = vec1[vec1_idx];
    T vec2_val = vec2[vec2_idx];
    T out_val;
    if constexpr (std::is_same_v<T, half>) {
        out_val = __hadd(__hmul(__hmul(vec1_val, vec2_val), __float2half(alpha)),
                         __hmul(__float2half(beta), in_val));
    } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
        float a = __bfloat162float(vec1_val), b = __bfloat162float(vec2_val), in = __bfloat162float(in_val);
        out_val = __float2bfloat16_rn(a * b * alpha + in * beta);
    } else {
        out_val = beta * in_val + alpha * vec1_val * vec2_val;
    }

    out[out_idx] = out_val;
    __syncthreads();
}
#endif
