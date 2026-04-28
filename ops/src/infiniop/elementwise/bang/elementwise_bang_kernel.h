#ifndef __INFINIOP_ELEMENTWISE_BANG_KERNEL_MLU__
#define __INFINIOP_ELEMENTWISE_BANG_KERNEL_MLU__

#include "../../devices/bang/bang_kernel_common.h"
#include "../../devices/bang/common_bang.h"

using namespace device::bang::kernel;

/**
 * @brief Core elementwise operation implementation for BANG device.
 *
 * @tparam N                Number of input tensors.
 * @tparam Op               Operator functor type.
 * @tparam Tdata            Data type for inputs and output.
 * @tparam Args             Additional arguments for operator.
 *
 * @param typed_inputs      Array of typed input pointers.
 * @param output            Output tensor pointer.
 * @param nram_buf          NRAM buffer for temporary storage.
 * @param input_indexes     Precomputed input indexes.
 * @param output_index      Starting output index.
 * @param num_elements      Number of elements to process.
 * @param output_contiguous Whether output is contiguous.
 * @param input_contiguous  Array indicating input contiguity.
 * @param ndim              Number of dimensions.
 * @param input_shape       Input shape in global memory.
 * @param input_strides     Input strides in global memory.
 * @param output_shape      Output shape in global memory.
 * @param output_strides    Output strides in global memory.
 * @param indexer           Input indexer helper.
 * @param start_idx         Starting index for this task.
 * @param args              Additional arguments for operator.
 */
template <size_t N, typename Op, typename Tdata, typename... Args>
__mlu_device__ void launchOp(
    Tdata **typed_inputs,
    Tdata *output,
    Tdata *nram_buf,
    size_t *input_indexes,
    size_t output_index,
    size_t num_elements,
    bool output_contiguous,
    const bool *input_contiguous,
    const bool *input_broadcasted,
    size_t ndim,
    const size_t *input_shapes,
    const ptrdiff_t *input_strides,
    const size_t *output_shape,
    const ptrdiff_t *output_strides,
    InputIndexer indexer,
    size_t start_idx,
    Args... args) {

    static_assert(N == Op::num_inputs, "template N is not equal to Op::num_inputs!");

    // NRAM memory planning
    const size_t nram_usable = NRAM_MAX_SIZE - (ALIGN_SIZE * (N + 1));
    const size_t max_batch = nram_usable / ((N + 1) * sizeof(Tdata));

    size_t processed = 0;
    while (processed < num_elements) {
        size_t curr_batch = std::min(max_batch, num_elements - processed);

        // Align memory address
        Tdata *aligned_buf = reinterpret_cast<Tdata *>(
            (reinterpret_cast<size_t>(nram_buf) + ALIGN_SIZE - 1) & ~(ALIGN_SIZE - 1));

        // 1. Copy input data to NRAM
        Tdata *input_buffers[N];
        for (size_t i = 0; i < N; ++i) {
            input_buffers[i] = aligned_buf + i * max_batch;

            if (input_contiguous[i]) {
                // Contiguous case - bulk copy
                __memcpy_async(input_buffers[i],
                               typed_inputs[i] + input_indexes[i] + processed,
                               curr_batch * sizeof(Tdata),
                               GDRAM2NRAM);
            } else {
                // Non-contiguous case - copy in contiguous chunks
                nonContiguousMemcpy<Tdata>(
                    input_buffers[i],
                    typed_inputs[i],
                    GDRAM2NRAM,
                    indexer,
                    i,
                    processed,
                    curr_batch,
                    start_idx,
                    ndim,
                    input_shapes + i * ndim,
                    input_strides + i * ndim,
                    true);
            }
        }
        __sync_io();

        // 2. Execute operation
        Tdata *output_buffer = aligned_buf + N * max_batch;
        Op op;
        op(output_buffer, input_buffers[0], input_buffers[1], curr_batch, args...);
        __sync_compute();

        // 3. Write back results
        if (output_contiguous) {
            // Contiguous output - bulk copy
            __memcpy_async(output + output_index + processed,
                           output_buffer,
                           curr_batch * sizeof(Tdata),
                           NRAM2GDRAM);
        } else {
            // Non-contiguous output - copy in contiguous chunks
            nonContiguousMemcpy<Tdata>(
                output,
                output_buffer,
                NRAM2GDRAM,
                indexer,
                0, // unused for output
                processed,
                curr_batch,
                start_idx,
                ndim,
                output_shape,
                output_strides,
                false);
        }

        processed += curr_batch;
    }
}

/**
 * @brief BANG kernel for elementwise operations.
 *
 * @tparam N        Number of input tensors.
 * @tparam Op       Operator functor type.
 * @tparam Tdata    Data type for inputs and output.
 * @tparam Args     Additional arguments for operator.
 *
 * @param output_size         Total output elements.
 * @param ndim                Number of dimensions.
 * @param output_contiguous   Whether output is contiguous.
 * @param input_contiguous    Input contiguity flags in global memory.
 * @param input_broadcasted   Input broadcast flags in global memory.
 * @param output_shape        Output shape in global memory.
 * @param input_shapes        Input shapes in global memory.
 * @param output_strides      Output strides in global memory.
 * @param input_strides       Input strides in global memory.
 * @param output              Output tensor pointer.
 * @param inputs              Array of input pointers.
 * @param args                Additional arguments for operator.
 */
template <size_t N, typename Op, typename Tdata, typename... Args>
__mlu_global__ void elementwiseKernel(
    size_t output_size,
    size_t ndim,
    bool output_contiguous,
    const bool *input_contiguous,
    const bool *input_broadcasted,
    const size_t *output_shape,
    const size_t *input_shapes,
    const ptrdiff_t *output_strides,
    const ptrdiff_t *input_strides,
    Tdata *output,
    const void *const *inputs,
    Args... args) {

    // Cast input pointers to the correct type
    Tdata *typed_inputs[N];
    for (size_t i = 0; i < N; ++i) {
        typed_inputs[i] = reinterpret_cast<Tdata *>(const_cast<void *>(inputs[i]));
    }

    // Calculate workload per task
    size_t elements_per_task = (output_size + taskDim - 1) / taskDim;
    size_t start_idx = taskId * elements_per_task;
    size_t end_idx = std::min(start_idx + elements_per_task, output_size);
    size_t num_elements = end_idx > start_idx ? end_idx - start_idx : 0;

    if (num_elements == 0) {
        return;
    }

    // Allocate NRAM buffer (shared by all inputs and output)
    __nram__ Tdata nram_buf[NRAM_MAX_SIZE / sizeof(Tdata)];

    // Get output index
    size_t output_index = getOutputIndex(start_idx, output_contiguous,
                                         ndim, output_shape, output_strides);

    // Create input indexer
    InputIndexer indexer{
        static_cast<size_t>(start_idx),
        ndim,
        input_contiguous,
        input_broadcasted,
        input_shapes,
        input_strides,
        output_strides};

    // Get index offsets for each operand
    size_t input_indexes[N];
    for (size_t i = 0; i < N; ++i) {
        input_indexes[i] = indexer(i, 0);
    }

    // Launch the operation with all required parameters
    launchOp<N, Op, Tdata>(typed_inputs, output, nram_buf, input_indexes,
                           output_index, num_elements, output_contiguous,
                           input_contiguous, input_broadcasted, ndim,
                           input_shapes, input_strides, output_shape,
                           output_strides, indexer, start_idx, args...);
}

/**
 * @brief Intermediate layer that determines optimal launch configuration before calling elementwiseKernel.
 *
 * @tparam N        Number of input tensors.
 * @tparam Op       Operator functor type.
 * @tparam Tdata    Data type for inputs and output.
 * @tparam Args     Additional arguments for operator.
 */
template <size_t N, typename Op, typename Tdata, typename... Args>
void launchElementwiseKernelWrapper(
    size_t output_size,
    size_t ndim,
    bool output_contiguous,
    const bool *input_contiguous,
    const bool *input_broadcasted,
    const size_t *output_shape,
    const size_t *input_shapes,
    const ptrdiff_t *output_strides,
    const ptrdiff_t *input_strides,
    Tdata *output,
    const void *const *inputs,
    cnrtQueue_t queue,
    const std::shared_ptr<device::bang::Handle::Internal> &internal,
    Args... args) {

    // Get hardware information from internal handle
    int core_per_cluster = internal->getCorePerCluster();
    int cluster_count = internal->getClusterCount();

    // Set kernel launch dimensions
    cnrtDim3_t dim;
    dim.x = core_per_cluster;
    dim.y = cluster_count;
    dim.z = 1;

    // Choose kernel type based on problem characteristics
    cnrtFunctionType_t func_type = cnrtFuncTypeBlock;
    if (output_size > 1024 * 1024 && output_contiguous) {
        // For large contiguous operations, use UNION type
        func_type = cnrtFuncTypeUnion1;
    }

    // Launch the kernel with optimal configuration
    elementwiseKernel<N, Op, Tdata><<<dim, func_type, queue>>>(
        output_size, ndim, output_contiguous,
        input_contiguous, input_broadcasted,
        output_shape, input_shapes,
        output_strides, input_strides,
        output, inputs, args...);
}

/**
 * @brief Macro for implementing elementwise kernel launch.
 *
 * @param OpName Name of the operation.
 * @param Op     Operator functor type.
 */
#define LAUNCH_ELEMENTWISE_KERNEL_IMPL(OpName, Op)                                \
    template <typename Tdata, typename... Args>                                   \
    void launch##OpName##Kernel(                                                  \
        size_t output_size,                                                       \
        size_t ndim,                                                              \
        bool output_contiguous,                                                   \
        const void *input_contiguous,                                             \
        const void *input_broadcasted,                                            \
        const void *output_shape,                                                 \
        const void *input_shapes,                                                 \
        const void *output_strides,                                               \
        const void *input_strides,                                                \
        void *output,                                                             \
        const void *const *inputs,                                                \
        cnrtQueue_t queue,                                                        \
        const std::shared_ptr<device::bang::Handle::Internal> &internal,          \
        Args... args) {                                                           \
        launchElementwiseKernelWrapper<Op::num_inputs, Op, Tdata>(                \
            output_size, ndim, output_contiguous,                                 \
            reinterpret_cast<const bool *>(input_contiguous),                     \
            reinterpret_cast<const bool *>(input_broadcasted),                    \
            reinterpret_cast<const size_t *>(output_shape),                       \
            reinterpret_cast<const size_t *>(input_shapes),                       \
            reinterpret_cast<const ptrdiff_t *>(output_strides),                  \
            reinterpret_cast<const ptrdiff_t *>(input_strides),                   \
            reinterpret_cast<Tdata *>(output), inputs, queue, internal, args...); \
    }

/**
 * @brief Macro for instantiating elementwise kernel for specific types.
 *
 * @param OpName Name of the operation.
 * @param T      Data type.
 * @param ...    Additional template arguments.
 */
/**
 * @brief Macro for instantiating elementwise kernel for specific types.
 *
 * @param OpName Name of the operation.
 * @param T      Data type.
 * @param ...    Additional template arguments.
 */
#define LAUNCH_ELEMENTWISE_KERNEL_INSTANTIATE(OpName, T, ...)            \
    template void launch##OpName##Kernel<T, ##__VA_ARGS__>(              \
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
        cnrtQueue_t queue,                                               \
        const std::shared_ptr<device::bang::Handle::Internal> &internal, \
        ##__VA_ARGS__);

#endif
