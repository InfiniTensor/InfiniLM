#ifndef __ALL_CUDA_H__
#define __ALL_CUDA_H__

template <size_t BLOCK_SIZE, typename Tdata>
__global__ void allReduceTempKernel(
    bool *temp_output,
    const Tdata *input,
    size_t input_size,
    size_t permuted_input_shape_size,
    size_t *permuted_input_shape,
    ptrdiff_t *permuted_input_strides) {
    __shared__ bool s_data[BLOCK_SIZE];
    size_t tid = threadIdx.x;
    size_t idx = tid + blockIdx.x * blockDim.x;
    if (idx < input_size) {
        size_t input_offset = indexToOffset(idx, permuted_input_shape_size, permuted_input_shape, permuted_input_strides);
        s_data[tid] = static_cast<bool>(input[input_offset]);
    } else {
        s_data[tid] = true;
    }
    __syncthreads();
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] = s_data[tid] && s_data[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        temp_output[blockIdx.x] = s_data[0];
    }
}

template <size_t BLOCK_SIZE>
__global__ void finalAllReduceKernel(
    bool *output,
    const bool *block_results,
    size_t num_blocks) {
    __shared__ bool s_data[BLOCK_SIZE];
    size_t tid = threadIdx.x;
    bool thread_val = true;
    for (size_t i = tid; i < num_blocks; i += blockDim.x) {
        thread_val = thread_val && block_results[i];
    }
    s_data[tid] = thread_val;
    __syncthreads();
    for (size_t s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] = s_data[tid] && s_data[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *output = s_data[0];
    }
}

template <size_t BLOCK_SIZE, typename Tdata>
__global__ void allKernel(
    bool *output,
    const Tdata *input,
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
    bool tempRes = true;
    for (size_t i = 0; i < reduce_num; i++) {
        size_t input_offset = indexToOffset(i + idx * reduce_num, permuted_input_shape_size, permuted_input_shape, permuted_input_strides);
        tempRes = tempRes && static_cast<bool>(input[input_offset]);
    }
    output[output_index] = tempRes;
}

#endif // __ALL_CUDA_H__
