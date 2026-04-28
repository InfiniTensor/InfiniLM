#ifndef __SUM_CUDA_H__
#define __SUM_CUDA_H__

template <size_t BLOCK_SIZE, typename Tdata, typename Tcompute>
__global__ void sumAllKernel(
    Tcompute *output,
    const Tdata *input,
    size_t input_size,
    size_t permuted_input_shape_size,
    size_t *permuted_input_shape,
    ptrdiff_t *permuted_input_strides) {
    __shared__ Tcompute s_data[BLOCK_SIZE];
    size_t tid = threadIdx.x;
    size_t idx = tid + blockIdx.x * blockDim.x;
    if (idx < input_size) {
        size_t input_offset = indexToOffset(idx, permuted_input_shape_size, permuted_input_shape, permuted_input_strides);
        s_data[tid] = static_cast<Tcompute>(input[input_offset]);
    } else {
        s_data[tid] = static_cast<Tcompute>(0.f);
    }
    __syncthreads();
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, s_data[0]);
    }
}

template <size_t BLOCK_SIZE, typename T>
__global__ void sumKernel(
    T *output,
    const T *input,
    size_t permuted_input_shape_size,
    size_t output_shape_size,
    size_t output_size,
    size_t reduce_num,
    size_t *permuted_input_shape,
    size_t *output_shape,
    ptrdiff_t *permuted_input_strides,
    ptrdiff_t *output_strides) {
    size_t tid = threadIdx.x;
    size_t idx = tid + blockIdx.x * blockDim.x;
    if (idx >= output_size) {
        return;
    }
    size_t output_index = indexToOffset(idx, output_shape_size, output_shape, output_strides);
    float tempSum = static_cast<float>(0.f);
    for (size_t i = 0; i < reduce_num; i++) {
        size_t input_offset = indexToOffset(i + idx * reduce_num, permuted_input_shape_size, permuted_input_shape, permuted_input_strides);
        tempSum += static_cast<float>(input[input_offset]);
    }
    output[output_index] = static_cast<T>(tempSum);
}

#endif // __SUM_CUDA_H__
