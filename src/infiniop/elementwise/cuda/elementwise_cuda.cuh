#ifndef __INFINIOP_ELEMENTWISE_CUDA_H__
#define __INFINIOP_ELEMENTWISE_CUDA_H__

#include "../../../utils.h"
#include "../../devices/cuda/cuda_common.cuh"
#include "../../devices/cuda/cuda_kernel_common.cuh"
#include "elementwise_cuda_api.cuh"

namespace op::elementwise::cuda {

/**
 * @brief Casts an untyped device pointer to a typed pointer of type T.
 *
 * @tparam T   Desired pointer type.
 *
 * @param ptr  Untyped pointer.
 * @return     Pointer of type const T*.
 */
template <typename T>
__device__ __forceinline__ const T *typedInputPtr(const void *ptr) {
    return reinterpret_cast<const T *>(ptr);
}

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
__device__ __forceinline__ size_t getOutputIndex(size_t idx, bool is_contiguous, size_t ndim,
                                                 const size_t *shape, const ptrdiff_t *strides) {
    return is_contiguous ? idx : device::cuda::indexToOffset(idx, ndim, shape, strides);
}

/**
 * @brief Computes input element offset for broadcasting and strided access.
 *
 * Used to map a linear output index to the corresponding index in an input tensor,
 * considering contiguity and broadcasting.
 */
struct InputIndexer {
    size_t idx;
    size_t ndim;
    const bool *input_contiguous;
    const bool *input_broadcasted;
    const size_t *input_shapes;
    const ptrdiff_t *input_strides;
    const ptrdiff_t *output_strides;

    /**
     * @brief Computes the memory offset for a given input tensor at current index.
     *
     * @param input_id  ID of the input tensor.
     * @return          Offset into the input tensor.
     */
    __device__ __forceinline__ size_t operator()(size_t input_id) const {
        return input_contiguous[input_id]
                 ? idx
                 : (input_broadcasted[input_id]
                        ? device::cuda::indexToReducedOffset(idx, ndim, output_strides, input_strides + input_id * ndim)
                        : device::cuda::indexToOffset(idx, ndim, input_shapes + input_id * ndim, input_strides + input_id * ndim));
    }
};

/**
 * @brief Invokes a callable with compile-time index constants.
 *
 * Used to unpack index sequence for variadic template processing of inputs.
 *
 * @tparam F    Callable type.
 * @tparam Is   Compile-time index sequence.
 *
 * @param f     Callable to invoke with index constants.
 */
template <typename F, size_t... Is>
__device__ __forceinline__ void unpackInputsAndApply(F &&f, std::index_sequence<Is...>) {
    f(std::integral_constant<size_t, Is>{}...);
}

/**
 * @brief CUDA kernel for performing elementwise operations on tensors where all inputs share the same data type.
 *
 * @tparam N        Number of input tensors.
 * @tparam Op       Operator type implementing operator()(Tdata...).
 * @tparam Tdata    Common data type for inputs and output.
 * @tparam Args     Additional arguments to pass to the operator.
 *
 * @param output_size         Total number of output elements.
 * @param ndim                Number of dimensions in tensors.
 * @param output_contiguous   Whether the output tensor is contiguous in memory.
 * @param input_contiguous    Array indicating if each input tensor is contiguous.
 * @param input_broadcasted   Array indicating if each input tensor is broadcasted.
 * @param output_shape        Shape of the output tensor.
 * @param input_shapes        Shapes of the input tensors.
 * @param output_strides      Strides for the output tensor.
 * @param input_strides       Strides for each input tensor.
 * @param output              Output buffer.
 * @param inputs              Array of input pointers, all of type Tdata.
 * @param offset              Linear offset to support partitioned execution.
 * @param args                Additional arguments passed to the operator.
 */
template <size_t N, typename Op, typename Tdata, typename... Args>
INFINIOP_CUDA_KERNEL elementwiseKernel(
    size_t output_size,
    size_t ndim,
    bool output_contiguous,
    const bool *__restrict__ input_contiguous,
    const bool *__restrict__ input_broadcasted,
    const size_t *__restrict__ output_shape,
    const size_t *__restrict__ input_shapes,
    const ptrdiff_t *__restrict__ output_strides,
    const ptrdiff_t *__restrict__ input_strides,
    Tdata *output,
    const void *const *inputs,
    size_t offset,
    Args... args) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset;

    if (idx < output_size) {
        const Tdata *const *typed_inputs = reinterpret_cast<const Tdata *const *>(inputs);
        size_t out_idx = getOutputIndex(idx, output_contiguous, ndim, output_shape, output_strides);
        InputIndexer indexer{idx, ndim, input_contiguous, input_broadcasted, input_shapes, input_strides, output_strides};

        unpackInputsAndApply(
            [&](auto... Is) {
                output[out_idx] = Op{}(typed_inputs[Is.value][indexer(Is.value)]..., std::forward<Args>(args)...);
            },
            std::make_index_sequence<N>{});
    }
}

/**
 * @brief CUDA kernel for performing an elementwise operation on tensors with support
 *        for broadcasting and mixed data types.
 *
 * @tparam Op     Operator type implementing a templated operator() for (Tout, Tin...).
 * @tparam Tout   Output data type.
 * @tparam Tin    Variadic input data types.
 *
 * @param output_size         Total number of output elements.
 * @param ndim                Number of dimensions in the tensors.
 * @param output_contiguous   Whether the output tensor is contiguous.
 * @param input_contiguous    Array indicating whether each input is contiguous.
 * @param input_broadcasted   Array indicating whether each input is broadcasted.
 * @param output_shape        Shape of the output tensor.
 * @param input_shapes        Shapes of the input tensors.
 * @param output_strides      Strides of the output tensor.
 * @param input_strides       Strides of the input tensors.
 * @param output              Pointer to the output buffer.
 * @param inputs              Array of untyped input pointers.
 * @param offset              Linear offset into the output for partitioned execution.
 */
template <typename Op, typename Tout, typename... Tin>
INFINIOP_CUDA_KERNEL elementwiseKernel(
    size_t output_size,
    size_t ndim,
    bool output_contiguous,
    const bool *__restrict__ input_contiguous,
    const bool *__restrict__ input_broadcasted,
    const size_t *__restrict__ output_shape,
    const size_t *__restrict__ input_shapes,
    const ptrdiff_t *__restrict__ output_strides,
    const ptrdiff_t *__restrict__ input_strides,
    Tout *output,
    const void *const *__restrict__ inputs,
    size_t offset) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset;

    if (idx < output_size) {
        size_t out_idx = getOutputIndex(idx, output_contiguous, ndim, output_shape, output_strides);
        InputIndexer indexer{idx, ndim, input_contiguous, input_broadcasted, input_shapes, input_strides, output_strides};

        unpackInputsAndApply(
            [&](auto... Is) {
                output[out_idx] = Op{}.template operator()<Tout, Tin...>(
                    (typedInputPtr<Tin>(inputs[Is.value])[indexer(Is.value)])...);
            },
            std::index_sequence_for<Tin...>{});
    }
}

struct DeviceImpl::Opaque {
    std::shared_ptr<device::cuda::Handle::Internal> internal;

    Opaque(const std::shared_ptr<device::cuda::Handle::Internal> &internal)
        : internal(internal) {}

    /**
     * @brief Executes an elementwise operation where all inputs and the output share the same data type.
     *
     * @tparam BLOCK_SIZE    CUDA block size used for kernel launch.
     * @tparam N             Number of input tensors.
     * @tparam Op            Functor representing the elementwise operation.
     * @tparam Tdata         Data type of both input and output tensors.
     * @tparam Args          Optional additional arguments passed to the operation.
     *
     * @param info           Metadata about the operation including shape, size, and dimensionality.
     * @param workspace      Temporary workspace used for storing metadata on device.
     * @param output         Pointer to the output buffer.
     * @param inputs         Vector of pointers to input buffers.
     * @param stream         CUDA stream for asynchronous execution.
     * @param args           Additional arguments forwarded to the operation.
     * @return infiniStatus_t Returns success or failure status.
     */
    template <uint32_t BLOCK_SIZE, size_t N, typename Op, typename Tdata, typename... Args>
    infiniStatus_t calculateImpl(const op::elementwise::ElementwiseInfo &info,
                                 void *workspace,
                                 void *output,
                                 const std::vector<const void *> &inputs,
                                 cudaStream_t stream,
                                 Args &&...args) {
        return launchElementwiseKernel<BLOCK_SIZE, N>(
            info, workspace,
            reinterpret_cast<Tdata *>(output), inputs,
            elementwiseKernel<N, Op, Tdata, Args...>,
            stream,
            std::forward<Args>(args)...);
    }

    /**
     * @brief Executes an elementwise operation with mixed input and output data types.
     *
     * @tparam BLOCK_SIZE    CUDA block size used for kernel launch.
     * @tparam N             Number of input tensors.
     * @tparam Op            Functor representing the elementwise operation.
     * @tparam Tout          Data type of the output tensor.
     * @tparam Tin...        Data types of the input tensors.
     * @tparam Args          Optional additional arguments passed to the operation.(UNUSED)
     *
     * @param info           Metadata about the operation including shape, size, and dimensionality.
     * @param workspace      Temporary workspace used for storing metadata on device.
     * @param output         Pointer to the output buffer.
     * @param inputs         Vector of pointers to input buffers.
     * @param stream         CUDA stream for asynchronous execution.
     * @param args           Additional arguments forwarded to the operation.
     * @return infiniStatus_t Returns success or failure status.
     */
    template <uint32_t BLOCK_SIZE, size_t N, typename Op, typename Tout, typename... Tin, typename... Args,
              std::enable_if_t<(sizeof...(Tin) == Op::num_inputs), int> = 0>
    infiniStatus_t calculateImpl(const op::elementwise::ElementwiseInfo &info,
                                 void *workspace,
                                 void *output,
                                 const std::vector<const void *> &inputs,
                                 cudaStream_t stream,
                                 Args &&...args) {
        return launchElementwiseKernel<BLOCK_SIZE, N>(
            info, workspace,
            reinterpret_cast<Tout *>(output), inputs,
            elementwiseKernel<Op, Tout, Tin...>,
            stream);
    }

private:
    /**
     * @brief Transfers elementwise operation metadata and input pointers from host to device memory.
     *
     * @tparam N                     Number of input tensors.
     *
     * @param info                   Elementwise operation metadata (shapes, strides, flags, etc.).
     * @param workspace              Pointer to device workspace memory for storing metadata and input pointers.
     * @param h_inputs_arr           Host array of input tensor pointers.
     * @param d_inputs_arr           Output reference to device array of input tensor pointers.
     * @param d_input_contiguous     Output reference to device array indicating whether each input is contiguous.
     * @param d_input_broadcasted    Output reference to device array indicating whether each input is broadcasted.
     * @param d_output_shape         Output reference to device array holding the output tensor shape.
     * @param d_output_strides       Output reference to device array holding output tensor strides.
     * @param d_input_shapes         Output reference to flattened input tensor shapes (N * ndim).
     * @param d_input_strides        Output reference to flattened input tensor strides (N * ndim).
     * @param stream                 CUDA stream used for asynchronous memory transfer.
     * @return infiniStatus_t        Status indicating success or failure of the memory transfer and setup.
     */
    template <size_t N>
    infiniStatus_t infoToDevice(
        const op::elementwise::ElementwiseInfo &info,
        void *workspace,
        const void *const *h_inputs_arr,
        const void **&d_inputs_arr,
        const bool *&d_input_contiguous,
        const bool *&d_input_broadcasted,
        const size_t *&d_output_shape,
        const ptrdiff_t *&d_output_strides,
        const size_t *&d_input_shapes,
        const ptrdiff_t *&d_input_strides,
        cudaStream_t stream) const {

        constexpr auto input_size = N;
        const auto ndim = info.getNdim();
        constexpr auto input_arr_size = N * sizeof(*h_inputs_arr);
        const int8_t *info_meta_start = info.getMetaStart();
        const int8_t *d_meta_start = reinterpret_cast<int8_t *>(workspace) + input_arr_size;

        // copy the input pointer array and meta to device
        CHECK_CUDA(cudaMemcpyAsync(workspace, h_inputs_arr, input_arr_size, cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync((void *)d_meta_start, info_meta_start, info.getMetaMemSize(), cudaMemcpyHostToDevice, stream));

        // offset/assign the pointers
        d_inputs_arr = reinterpret_cast<const void **>(workspace);
        d_output_shape = reinterpret_cast<const size_t *>(d_meta_start);
        d_output_strides = reinterpret_cast<const ptrdiff_t *>(d_output_shape + ndim);
        d_input_shapes = reinterpret_cast<const size_t *>(d_output_strides + ndim);
        d_input_strides = reinterpret_cast<const ptrdiff_t *>(d_input_shapes + input_size * ndim);
        d_input_contiguous = reinterpret_cast<const bool *>(d_input_strides + input_size * ndim);
        d_input_broadcasted = reinterpret_cast<const bool *>(d_input_contiguous + input_size);

        return INFINI_STATUS_SUCCESS;
    }

    /**
     * @brief Launches the elementwise kernel for the specified operation.
     *
     * @tparam BLOCK_SIZE   Number of threads per block.
     * @tparam N            Number of input tensors.
     * @tparam KernelFunc   Type of the kernel function pointer.
     * @tparam Tout         Output data type.
     * @tparam Args         Additional arguments to be forwarded to the kernel.
     *
     * @param info          Metadata about the elementwise operation (shapes, strides, etc.).
     * @param workspace     CUDA memory used for storing metadata.
     * @param output        Pointer to output buffer on device.
     * @param inputs        Vector of device pointers to input tensors.
     * @param kernel_func   Kernel function to launch.
     * @param stream        CUDA stream for asynchronous execution.
     * @param args          Additional arguments passed to the kernel.
     * @return infiniStatus_t  Status code indicating success or failure.
     */
    template <uint32_t BLOCK_SIZE, size_t N, typename KernelFunc, typename Tout, typename... Args>
    infiniStatus_t launchElementwiseKernel(
        const op::elementwise::ElementwiseInfo &info,
        void *workspace,
        Tout *output,
        const std::vector<const void *> &inputs,
        KernelFunc kernel_func,
        cudaStream_t stream,
        Args &&...args) {

        auto output_size = info.getOutputSize();
        if (output_size == 0) {
            return INFINI_STATUS_SUCCESS;
        }

        // Device pointers
        const void **d_inputs_arr = nullptr;
        const bool *d_input_contiguous = nullptr;
        const bool *d_input_broadcasted = nullptr;
        const size_t *d_output_shape = nullptr;
        const ptrdiff_t *d_output_strides = nullptr;
        const size_t *d_input_shapes = nullptr;
        const ptrdiff_t *d_input_strides = nullptr;

        CHECK_STATUS(infoToDevice<N>(info, workspace, inputs.data(), d_inputs_arr,
                                     d_input_contiguous, d_input_broadcasted,
                                     d_output_shape, d_output_strides,
                                     d_input_shapes, d_input_strides, stream));

        dim3 blockDims(std::min(BLOCK_SIZE, static_cast<uint32_t>(internal->maxThreadsPerBlock())));
        dim3 gridDims(std::min(uint32_t(CEIL_DIV(output_size, blockDims.x)), static_cast<uint32_t>(internal->gridSizeX())));
        size_t step = gridDims.x * blockDims.x;

        for (size_t i = 0; i < output_size; i += step) {
            kernel_func<<<gridDims, blockDims, 0, stream>>>(
                output_size, info.getNdim(), info.isOutputContiguous(),
                d_input_contiguous, d_input_broadcasted,
                d_output_shape, d_input_shapes,
                d_output_strides, d_input_strides,
                output, reinterpret_cast<const void **>(d_inputs_arr),
                i, std::forward<Args>(args)...);
        }

        return INFINI_STATUS_SUCCESS;
    }
};

template <typename... Args>
utils::Result<DeviceImpl *> DeviceImpl::create(Args &&...args) {
    auto opaque = std::make_shared<Opaque>(std::forward<Args>(args)...);
    return utils::Result<DeviceImpl *>(new DeviceImpl(opaque));
}

/* Invoke elementwise operation for different input types */
template <unsigned int BLOCK_SIZE, typename Op, typename Tout, typename... Tin, typename... Args,
          std::enable_if_t<(sizeof...(Tin) == Op::num_inputs), int>>
infiniStatus_t DeviceImpl::calculate(const op::elementwise::ElementwiseInfo &info,
                                     void *workspace,
                                     void *output,
                                     const std::vector<const void *> &inputs,
                                     void *stream,
                                     Args &&...args) {
    constexpr size_t N = Op::num_inputs;
    static_assert(sizeof...(Tin) == N, "Input type count mismatch");
    return _opaque->calculateImpl<BLOCK_SIZE, N, Op, Tout, Tin...>(
        info, workspace, output, inputs,
        reinterpret_cast<cudaStream_t>(stream),
        std::forward<Args>(args)...);
}

/* Invoke elementwise operation when all inputs have the same dtype */
template <unsigned int BLOCK_SIZE, typename Op, typename Tdata, typename... Args>
infiniStatus_t DeviceImpl::calculate(const op::elementwise::ElementwiseInfo &info,
                                     void *workspace,
                                     void *output,
                                     const std::vector<const void *> &inputs,
                                     void *stream,
                                     Args &&...args) {
    constexpr size_t N = Op::num_inputs;
    return _opaque->calculateImpl<BLOCK_SIZE, N, Op, Tdata>(
        info, workspace, output, inputs,
        reinterpret_cast<cudaStream_t>(stream),
        std::forward<Args>(args)...);
}

} // namespace op::elementwise::cuda

#endif // __INFINIOP_ELEMENTWISE_CUDA_H__
