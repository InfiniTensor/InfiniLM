#ifndef __MASKED_SELECT_MOORE_KERNEL_CUH__
#define __MASKED_SELECT_MOORE_KERNEL_CUH__

template <size_t BLOCK_SIZE>
INFINIOP_MOORE_KERNEL maskedSelectGetMarkScanOnceKernel(
    const bool *mask, size_t *mark_scan, size_t total_elements,
    size_t *shape, ptrdiff_t *mask_strides, size_t ndim) {

    __shared__ __align__(128) size_t smem[BLOCK_SIZE];

    size_t tid = threadIdx.x;
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < total_elements) {
        size_t mask_offset = device::moore::indexToOffset(index, ndim, shape, mask_strides);
        smem[tid] = mask[mask_offset];
    } else {
        smem[tid] = 0;
    }

    for (int s = 1; s < BLOCK_SIZE; s <<= 1) {
        __syncthreads();

        float temp;
        if (tid >= s) {
            temp = smem[tid] + smem[tid - s];
        }
        __syncthreads();

        if (tid >= s) {
            smem[tid] = temp;
        }
    }

    if (index < total_elements) {
        mark_scan[index] = smem[tid];
    }
}

template <size_t BLOCK_SIZE>
INFINIOP_MOORE_KERNEL maskedSelectScanWithStrideKernel(
    size_t *mark_scan, size_t total_elements, size_t stride) {

    __shared__ __align__(128) size_t smem[BLOCK_SIZE];

    size_t tid = threadIdx.x;
    size_t index = (blockDim.x * blockIdx.x + threadIdx.x + 1) * stride - 1;
    smem[tid] = index < total_elements ? mark_scan[index] : 0.;

    for (int s = 1; s < BLOCK_SIZE; s <<= 1) {
        __syncthreads();

        size_t temp;
        if (tid >= s) {
            temp = smem[tid] + smem[tid - s];
        }
        __syncthreads();

        if (tid >= s) {
            smem[tid] = temp;
        }
    }

    if (index < total_elements) {
        mark_scan[index] = smem[tid];
    }
}

template <size_t BLOCK_SIZE>
INFINIOP_MOORE_KERNEL maskedSelectCountScanResultKernel(
    size_t *mark_scan, size_t *scan_result, size_t total_elements) {

    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= total_elements) {
        return;
    }

    size_t val = mark_scan[index];

    for (size_t s = BLOCK_SIZE; s < total_elements; s *= BLOCK_SIZE) {
        size_t now_stride_block = (index + 1) / s;
        size_t pre_stride_block = now_stride_block * s - 1;
        if (now_stride_block == 0 || (index + 1) % s == 0 || (pre_stride_block + 1) % (s * BLOCK_SIZE) == 0) {
            continue;
        }
        val += mark_scan[pre_stride_block];
    }

    scan_result[index] = val;
}

template <size_t BLOCK_SIZE, typename T>
INFINIOP_MOORE_KERNEL maskedSelectGetDataKernel(
    const T *input, const bool *mask, size_t *scan_result, T *data, size_t total_elements,
    size_t *shape, ptrdiff_t *input_strides, ptrdiff_t *mask_strides, size_t ndim) {

    size_t index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index >= total_elements) {
        return;
    }

    size_t input_offset = device::moore::indexToOffset(index, ndim, shape, input_strides);
    size_t mask_offset = device::moore::indexToOffset(index, ndim, shape, mask_strides);

    if (mask[mask_offset]) {
        data[scan_result[index] - 1] = input[input_offset];
    }
}

#endif
