#ifndef __INNER_KERNEL_CUH__
#define __INNER_KERNEL_CUH__

template <size_t BLOCK_SIZE, typename T>
INFINIOP_CUDA_KERNEL innerKernel(
    const T *input, const T *other, T *out,
    size_t total_elements, size_t oper_len, size_t *out_shape,
    ptrdiff_t *input_strides, ptrdiff_t *other_strides, ptrdiff_t *out_strides,
    size_t input_ndim, size_t other_ndim, size_t out_ndim) {

    size_t out_index = blockDim.x * blockIdx.x + threadIdx.x;
    if (out_index >= total_elements) {
        return;
    }

    size_t out_offset = device::nvidia::indexToOffset(out_index, out_ndim, out_shape, out_strides);

    size_t out_dim_pos = out_ndim - 1;
    size_t index = out_index;

    ptrdiff_t input_offset = 0;
    ptrdiff_t other_offset = 0;

    for (int i = (int)other_ndim - 2; i >= 0; i--) {
        other_offset += (index % out_shape[out_dim_pos]) * other_strides[i];
        index /= out_shape[out_dim_pos--];
    }
    for (int i = (int)input_ndim - 2; i >= 0; i--) {
        input_offset += (index % out_shape[out_dim_pos]) * input_strides[i];
        index /= out_shape[out_dim_pos--];
    }

    T tmp = 0;
    for (size_t i = 0; i < oper_len; i++) {
        tmp += input[input_offset] * other[other_offset];
        input_offset += input_strides[input_ndim - 1];
        other_offset += other_strides[other_ndim - 1];
    }

    out[out_offset] = tmp;
}

#endif // __INNER_KERNEL_CUH__
