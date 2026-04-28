#ifndef __INFINIOP_BANG_KERNEL_COMMON_H__
#define __INFINIOP_BANG_KERNEL_COMMON_H__

// Include Cambricon CNNL and CNRT headers for MLU (Machine Learning Unit) specific functions
#include "cnnl.h"
#include "cnrt.h"

namespace device::bang::kernel {

template <typename T>
__mlu_device__ float to_float(const T &v) {
    return static_cast<float>(v);
}

template <typename T>
__mlu_device__ bfloat16_t to_bfloat16(const T &v) {
    return static_cast<bfloat16_t>(v);
}

template <typename T>
__mlu_device__ half to_half(const T &v) {
    return static_cast<half>(v);
}

/**
 * @brief Converts a flattened index to a memory offset considering tensor striding.
 *
 * This is the general case for non-contiguous tensors where elements are not
 * stored sequentially in memory.
 *
 * @param flat_index The flattened index in the tensor
 * @param ndim Number of dimensions
 * @param shape Tensor shape
 * @param strides Tensor strides (in elements)
 * @return size_t Offset in the tensor's memory
 */
inline __mlu_device__ size_t indexToOffset(
    size_t flat_index,
    size_t ndim,
    const size_t *shape,
    const ptrdiff_t *strides) {

    size_t res = 0;
    // Process dimensions from highest to lowest
    for (size_t i = ndim; i-- > 0;) {
        // Add contribution from this dimension
        res += (flat_index % shape[i]) * strides[i];
        // Remove the contribution from this dimension
        flat_index /= shape[i];
    }
    return res;
}

/**
 * @brief Helper struct for computing input tensor indices considering broadcasting and striding.
 *
 * This is particularly useful for operations where inputs may be broadcasted
 * to match the output shape, or may have non-contiguous memory layouts.
 */
struct InputIndexer {
    size_t idx;                      // Base index for this task
    size_t ndim;                     // Number of dimensions
    const bool *input_contiguous;    // Array indicating which inputs are contiguous
    const bool *input_broadcasted;   // Array indicating which inputs are broadcasted
    const size_t *input_shapes;      // Array of input shapes (concatenated)
    const ptrdiff_t *input_strides;  // Array of input strides (concatenated)
    const ptrdiff_t *output_strides; // Output tensor strides

    /**
     * @brief Computes memory offset for input tensor element.
     *
     * @param input_id    Input tensor ID.
     * @param element_idx Element index in output tensor.
     * @return size_t     Memory offset in input tensor.
     */
    __mlu_device__ size_t operator()(size_t input_id, size_t element_idx) const {
        size_t global_idx = idx + element_idx;
        return input_contiguous[input_id]
                 ? global_idx // Simple case: contiguous memory
                 : indexToOffset(global_idx, ndim, input_shapes + input_id * ndim, input_strides + input_id * ndim);
    }
};

/**
 * @brief Computes output tensor index considering striding.
 *
 * @param idx            Linear index.
 * @param is_contiguous  Whether output is contiguous.
 * @param ndim           Number of dimensions.
 * @param shape          Output tensor shape.
 * @param strides        Output tensor strides.
 * @return size_t        Memory offset in output tensor.
 */
inline __mlu_device__ size_t
getOutputIndex(size_t idx,
               bool is_contiguous,
               size_t ndim,
               const size_t *shape,
               const ptrdiff_t *strides) {
    return is_contiguous ? idx : indexToOffset(idx, ndim, shape, strides);
} /**
   * @brief Calculates optimal chunk size for memory operations based on tensor contiguity.
   *
   *        This function doesn't handle tensors with non-standard strides, which
   *        require more general optimizations not specific to Cambricon.
   *
   * @param global_idx_    Starting global index.
   * @param ndim           Number of dimensions.
   * @param shape          Tensor shape.
   * @param strides        Tensor strides.
   * @param max_len        Maximum allowed chunk size.
   * @return size_t        Optimal chunk size for memory operations.
   */
__mlu_device__ size_t calculateChunkSize(
    size_t global_idx_,
    size_t ndim,
    const size_t *shape,
    const ptrdiff_t *strides,
    size_t max_len) {
    // Find the last dimension that is contiguous
    int last_contiguous_dim = -1;
    ptrdiff_t expected_stride = 1;

    for (int i = (int)ndim - 1; i >= 0; --i) {
        if (strides[i] != expected_stride) {
            break;
        }
        last_contiguous_dim = i;
        if (i > 0) {
            expected_stride *= shape[i];
        }
    }

    if (last_contiguous_dim < 0) {
        return 1;
    }

    // Calculate position in the contiguous block
    size_t global_idx = global_idx_;
    size_t pos_in_block = 0;
    size_t block_size = 1;

    for (int i = (int)ndim - 1; i >= last_contiguous_dim; --i) {
        size_t dim_idx = global_idx % shape[i];
        pos_in_block += dim_idx * block_size;
        block_size *= shape[i];
        global_idx /= shape[i];
    }

    size_t remaining_in_block = block_size - pos_in_block;
    return std::min(max_len, remaining_in_block);
}

/**
 * @brief Helper function for non-contiguous memory copy
 *
 * @param dst Destination buffer
 * @param src Source buffer
 * @param direction Memory copy direction (GDRAM2NRAM or NRAM2GDRAM)
 * @param indexer Input indexer helper (for input copies)
 * @param input_idx Input tensor index (for input copies)
 * @param processed Number of elements already processed
 * @param curr_batch Current batch size
 * @param start_idx Starting index for this task
 * @param ndim Number of dimensions
 * @param shape Tensor shape
 * @param strides Tensor strides
 * @param is_input_copy Whether this is an input copy operation
 */
template <typename Tdata>
__mlu_device__ void nonContiguousMemcpy(
    Tdata *dst,
    Tdata *src,
    mluMemcpyDirection_t direction,
    InputIndexer &indexer,
    size_t input_idx,
    size_t processed,
    size_t curr_batch,
    size_t start_idx,
    size_t ndim,
    const size_t *shape,
    const ptrdiff_t *strides,
    bool is_input_copy) {

    size_t remaining = curr_batch;
    size_t current_pos = 0;

    while (remaining > 0) {
        size_t element_offset = is_input_copy ? indexer(input_idx, processed + current_pos) : getOutputIndex(start_idx + processed + current_pos,
                                                                                                             false, // output_contiguous is false for non-contiguous
                                                                                                             ndim, shape, strides);

        size_t chunk_size = calculateChunkSize(start_idx + processed + current_pos,
                                               ndim,
                                               shape,
                                               strides,
                                               remaining);

        __memcpy_async(dst + (is_input_copy ? current_pos : element_offset),
                       src + (is_input_copy ? element_offset : current_pos),
                       chunk_size * sizeof(Tdata),
                       direction);

        current_pos += chunk_size;
        remaining -= chunk_size;
    }
}

} // namespace device::bang::kernel

#endif
