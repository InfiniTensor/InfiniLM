#ifndef __INFINIOP_ELEMENTWISE_KUNLUN_XPU__
#define __INFINIOP_ELEMENTWISE_KUNLUN_XPU__

#include "../../devices/kunlun/kunlun_kernel_common.h"

using namespace device::kunlun::kernel;

/**
 * @brief Computes input tile offset
 */
struct InputIndexer {
    size_t idx;
    size_t ndim;
    const bool *input_contiguous;
    const bool *input_broadcasted;
    const _size_t *input_shapes;
    const _ptrdiff_t *input_strides;
    const _ptrdiff_t *output_strides;

    __device__ size_t operator()(size_t input_id) const {
        return input_contiguous[input_id]
                 ? idx
                 : (input_broadcasted[input_id]
                        ? indexToReducedOffset(idx, ndim, output_strides, input_strides + input_id * ndim)
                        : indexToOffset(idx, ndim, input_shapes + input_id * ndim, input_strides + input_id * ndim));
    }
};

/**
 * @brief Computes the output index in memory, accounting for strides if non-contiguous.
 *
 * @param idx            Linear index.
 * @param is_contiguous  Whether the output tensor is contiguous.
 * @param ndim           Number of dimensions.
 * @param shape          Shape of the output tensor.
 * @param strides        Strides of the output tensor.
 * @return               Memory offset index.
 */
inline __device__ size_t
getOutputIndex(size_t idx,
               bool is_contiguous,
               size_t ndim,
               const _size_t *shape,
               const _ptrdiff_t *strides) {
    return is_contiguous ? idx : indexToOffset(idx, ndim, shape, strides);
}

template <size_t N, typename Op, typename Tdata, typename... Args>
__device__ void launchOp(
    __global_ptr__ Tdata **typed_inputs, // gm pointer
    __global_ptr__ Tdata *output,        // gm pointer output
    Tdata *inputs_buf,                   // local mem buffer
    size_t *input_indexes,
    size_t output_index,
    Args... args) {

    static_assert(N == Op::num_inputs, "template N is not equal to Op::num_inputs!\n");

#pragma unroll
    // Copy inputs to buf
    for (size_t i = 0; i < N; i++) {
        auto gm = typed_inputs[i] + input_indexes[i];
        auto lm = inputs_buf + i;
        GM2LM_ASYNC(gm, lm, 1 * sizeof(Tdata));
    }
    mfence();

    // Calculate elementwise
    // Inputs save all operands
    Tdata out = Op{}(inputs_buf, args...);

    // Copy out to gm
    LM2GM_ASYNC(&out, output + output_index, 1 * sizeof(Tdata));
    mfence();
}

template <size_t N, typename Op, typename Tdata, typename... Args>
__global__ void elementwiseKernel(
    size_t output_size,
    size_t ndim,
    bool output_contiguous,
    const bool *input_contiguous_gm,
    const bool *input_broadcasted_gm,
    const _size_t *output_shape_gm,
    const _size_t *input_shapes_gm,
    const _ptrdiff_t *output_strides_gm,
    const _ptrdiff_t *input_strides_gm,
    Tdata *output,
    const void *const *inputs,
    Args... args) {

    int cid = core_id();
    int ncores = core_num();
    if (cid >= ncores) {
        return;
    }
    int thread_id = ncores * cluster_id() + cid;
    int nthreads = ncores * cluster_num();

    // Cast input gm pointer type
    auto typed_inputs = reinterpret_cast<const __global_ptr__ Tdata *const __global_ptr__ *>(inputs);

    const int BUFF_SIZE = 64;
    // Input data cache
    __local__ Tdata inputs_buf[N];
    // Input contiguous/broadcasted flags
    __local__ bool input_contiguous[N];
    __local__ bool input_broadcasted[N];
    // Input shape/strides
    __local__ _size_t input_shapes[N * ndim];
    __local__ _ptrdiff_t input_strides[N * ndim];
    // Output shape/strides
    __local__ _size_t output_shape[ndim];
    __local__ _ptrdiff_t output_strides[ndim];
    // Inputs gm ptr buf
    __local__ __global_ptr__ Tdata *typed_inputs_ptr[N];

    // Load from gm
    GM2LM_ASYNC(input_contiguous_gm, input_contiguous, N * sizeof(bool));
    GM2LM_ASYNC(input_broadcasted_gm, input_broadcasted, N * sizeof(bool));
    GM2LM_ASYNC(input_shapes_gm, input_shapes, N * ndim * sizeof(_size_t));
    GM2LM_ASYNC(input_strides_gm, input_strides, N * ndim * sizeof(_ptrdiff_t));
    GM2LM_ASYNC(output_shape_gm, output_shape, ndim * sizeof(_size_t));
    GM2LM_ASYNC(output_strides_gm, output_strides, ndim * sizeof(_ptrdiff_t));
    GM2LM_ASYNC(typed_inputs, typed_inputs_ptr, N * sizeof(__global_ptr__ Tdata *));
    mfence();

    int len_per_loop = min(BUFF_SIZE, roundup_div(output_size, nthreads));

    for (int start = thread_id * len_per_loop; start < output_size; start += nthreads * len_per_loop) {
        size_t read_len = min(len_per_loop, output_size - start);
        for (int idx = start; idx < start + read_len; ++idx) {
            size_t out_idx = getOutputIndex(static_cast<size_t>(idx), output_contiguous,
                                            ndim, output_shape, output_strides);
            InputIndexer indexer{static_cast<size_t>(idx), ndim, input_contiguous, input_broadcasted,
                                 input_shapes, input_strides, output_strides};
            // Get index offset for every operand
            size_t indexes[N];
            for (size_t i = 0; i < N; i++) {
                indexes[i] = indexer(i);
            }
            // Launch operater
            launchOp<N, Op, Tdata>(&typed_inputs_ptr[0], output, inputs_buf, indexes, out_idx, args...);
        }
    }
    sync_cluster();
}

#define LAUNCH_ELEMENTWISE_KERNEL_IMPL(OpName, Op)                       \
    template <typename Tdata, typename... Args>                          \
    void launch##OpName##Kernel(                                         \
        size_t output_size,                                              \
        size_t ndim,                                                     \
        bool output_contiguous,                                          \
        const void *input_contiguous,                                    \
        const void *input_broadcasted,                                   \
        const void *output_shape,                                        \
        const void *input_shapes,                                        \
        const void *output_strides,                                      \
        const void *input_strides,                                       \
        void *output,                                                    \
        const void *const *inputs,                                       \
        XPUStream stream,                                                \
        Args... args) {                                                  \
        elementwiseKernel<Op::num_inputs, Op, Tdata><<<8, 64, stream>>>( \
            output_size, ndim, output_contiguous,                        \
            reinterpret_cast<const bool *>(input_contiguous),            \
            reinterpret_cast<const bool *>(input_broadcasted),           \
            reinterpret_cast<const _size_t *>(output_shape),             \
            reinterpret_cast<const _size_t *>(input_shapes),             \
            reinterpret_cast<const _ptrdiff_t *>(output_strides),        \
            reinterpret_cast<const _ptrdiff_t *>(input_strides),         \
            reinterpret_cast<Tdata *>(output), inputs, args...);         \
    }

#define LAUNCH_ELEMENTWISE_KERNEL_INSTANTIATE(OpName, T, ...) \
    template void launch##OpName##Kernel<T, ##__VA_ARGS__>(   \
        size_t output_size,                                   \
        size_t ndim,                                          \
        bool output_contiguous,                               \
        const void *input_contiguous,                         \
        const void *input_broadcasted,                        \
        const void *output_shape,                             \
        const void *input_shapes,                             \
        const void *output_strides,                           \
        const void *input_strides,                            \
        void *output,                                         \
        const void *const *inputs,                            \
        XPUStream stream,                                     \
        ##__VA_ARGS__);

#endif
