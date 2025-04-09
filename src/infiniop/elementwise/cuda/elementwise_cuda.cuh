#ifndef __INFINIOP_ELEMENTWISE_CUDA_H__
#define __INFINIOP_ELEMENTWISE_CUDA_H__

#include "../../../utils.h"
#include "../../devices/cuda/cuda_common.cuh"
#include "elementwise_cuda_api.cuh"

namespace op::elementwise::cuda {

/**
 * @brief Helper device function to expand a compile-time index sequence into individual constants
 *        and pass them to a lambda.
 *
 * @tparam Lambda  Type of the lambda function to invoke.
 * @tparam Is      Index sequence values (automatically deduced).
 * @param lambda   Lambda to be called with std::integral_constant<size_t, Is>... as arguments.
 */
template <typename Lambda, size_t... Is>
__device__ __forceinline__ void call_expand(Lambda lambda, std::index_sequence<Is...>) {
    lambda(std::integral_constant<size_t, Is>{}...);
}

/**
 * @brief CUDA kernel for performing elementwise operations on tensors where all inputs share the same data type.
 *
 * @tparam Op       Operator type implementing operator()(Tdata...).
 * @tparam Tdata    Common data type for inputs and output.
 * @tparam N        Number of input tensors.
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
 * @param input_size          Total number of input elements (optional, may be unused).
 * @param output              Output buffer.
 * @param inputs              Array of input pointers, all of type Tdata.
 * @param offset              Linear offset to support partitioned execution.
 * @param args                Additional arguments passed to the operator.
 */
template <size_t N, typename Op, typename Tdata, typename... Args>
INFINIOP_CUDA_KERNEL elementwise_kernel(
    size_t output_size,
    size_t ndim,
    bool output_contiguous,
    const bool *__restrict__ input_contiguous,
    const bool *__restrict__ input_broadcasted,
    const size_t *__restrict__ output_shape,
    const size_t *__restrict__ *__restrict__ input_shapes,
    const ptrdiff_t *__restrict__ output_strides,
    const ptrdiff_t *__restrict__ *__restrict__ input_strides,
    size_t input_size,
    Tdata *output,
    const Tdata *const *inputs,
    size_t offset,
    Args... args) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (idx < output_size) {
        size_t out_idx = output_contiguous ? idx
                                           : device::cuda::indexToOffset(idx, ndim, output_shape, output_strides);

        auto get_input_idx = [&] __device__(size_t input_id) {
            return input_contiguous[input_id] ? idx
                                              : (input_broadcasted[input_id]
                                                     ? device::cuda::indexToReducedOffset(idx, ndim, output_strides, input_strides[input_id])
                                                     : device::cuda::indexToOffset(idx, ndim, input_shapes[input_id], input_strides[input_id]));
        };

        // Use a helper to expand the index sequence into individual compile-time constants
        auto expand_inputs = [&] __device__(auto... idxs) {
            if constexpr (std::is_same_v<Tdata, fp16_t>) {
                output[out_idx] = utils::cast<fp16_t>(
                    Op{}(utils::cast<float>(inputs[idxs.value][get_input_idx(idxs.value)])...,
                         std::forward<Args>(args)...));
            } else {
                output[out_idx] = Op{}(
                    inputs[idxs.value][get_input_idx(idxs.value)]...,
                    std::forward<Args>(args)...);
            }
        };

        call_expand(expand_inputs, std::make_index_sequence<N>{});
    }
}

/**
 * @brief Casts an untyped device pointer to a typed pointer of type T.
 *
 * @tparam T   Desired pointer type.
 * @param ptr  Untyped pointer.
 * @return     Pointer of type const T*.
 */
template <typename T>
__device__ inline const T *typed_input_ptr(const void *ptr) {
    return reinterpret_cast<const T *>(ptr);
}

/**
 * @brief Launches a type-safe elementwise operation on a single output element.
 *
 * @tparam Op     Operator type implementing a templated operator() for (Tout, Tin...).
 * @tparam Tout   Output data type.
 * @tparam Tin    Variadic input data types.
 * @tparam Is     Index sequence corresponding to each input.
 *
 * @param idx                 Linear index in the flattened output space.
 * @param out_idx             Actual output index (may be non-contiguous).
 * @param ndim                Number of dimensions in the tensors.
 * @param input_contiguous    Array indicating whether each input is contiguous.
 * @param input_broadcasted   Array indicating whether each input is broadcasted.
 * @param input_shapes        Shapes of the input tensors.
 * @param input_strides       Strides of the input tensors.
 * @param inputs              Raw pointers to input data.
 * @param output              Pointer to output data.
 * @param ...                 Index sequence used for unpacking variadic inputs.
 */
template <typename Op, typename Tout, typename... Tin, size_t... Is>
__device__ void launch_op(
    size_t idx,
    size_t out_idx,
    size_t ndim,
    const bool *__restrict__ input_contiguous,
    const bool *__restrict__ input_broadcasted,
    const size_t *__restrict__ const *__restrict__ input_shapes,
    const ptrdiff_t *__restrict__ const *__restrict__ input_strides,
    const ptrdiff_t *__restrict__ output_strides,
    const void *const *__restrict__ inputs,
    Tout *output,
    std::index_sequence<Is...>) {

    auto get_input_idx = [&] __device__(size_t input_id) {
        return input_contiguous[input_id]
                 ? idx
                 : (input_broadcasted[input_id]
                        ? device::cuda::indexToReducedOffset(idx, ndim, output_strides, input_strides[input_id])
                        : device::cuda::indexToOffset(idx, ndim, input_shapes[input_id], input_strides[input_id]));
    };

    output[out_idx] = Op{}.template operator()<Tout, Tin...>(
        (typed_input_ptr<Tin>(inputs[Is])[get_input_idx(Is)])...);
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
 * @param input_size          Total number of input elements (unused here, but may be used for validation).
 * @param output              Pointer to the output buffer.
 * @param inputs              Array of untyped input pointers.
 * @param offset              Linear offset into the output for partitioned execution.
 */
template <typename Op, typename Tout, typename... Tin>
INFINIOP_CUDA_KERNEL elementwise_kernel(
    size_t output_size,
    size_t ndim,
    bool output_contiguous,
    const bool *__restrict__ input_contiguous,
    const bool *__restrict__ input_broadcasted,
    const size_t *__restrict__ output_shape,
    const size_t *__restrict__ const *__restrict__ input_shapes,
    const ptrdiff_t *__restrict__ output_strides,
    const ptrdiff_t *__restrict__ const *__restrict__ input_strides,
    size_t input_size,
    Tout *output,
    const void *const *__restrict__ inputs,
    size_t offset) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (idx >= output_size) {
        return;
    }

    size_t out_idx = output_contiguous
                       ? idx
                       : device::cuda::indexToOffset(idx, ndim, output_shape, output_strides);

    launch_op<Op, Tout, Tin...>(
        idx,
        out_idx,
        ndim,
        input_contiguous,
        input_broadcasted,
        input_shapes,
        input_strides,
        output_strides,
        inputs,
        output,
        std::index_sequence_for<Tin...>{});
}

struct DeviceImpl::Opaque {
    std::shared_ptr<device::cuda::Handle::Internal> internal;

    Opaque(const std::shared_ptr<device::cuda::Handle::Internal> &internal)
        : internal(internal) {}

    /**
     * @brief Performs elementwise operations when all inputs and the output share the same data type.
     *
     * @tparam BLOCK_SIZE  The block size for the kernel launch.
     * @tparam N           The number of input tensors.
     * @tparam Op          The operation to perform (e.g., addition, multiplication).
     * @tparam Tdata       The data type of the input and output tensors.
     * @tparam Args        Additional arguments to be passed to the operation.
     * @param info         Structure containing elementwise operation information (size, shape, etc.).
     * @param output       Pointer to the output memory where results will be stored.
     * @param inputs       Vector of pointers to input tensors.
     * @param stream       CUDA stream used for asynchronous execution.
     * @param args         Additional arguments for the operation.
     * @return infiniStatus_t  Status indicating success or failure.
     */
    template <size_t BLOCK_SIZE, size_t N, typename Op, typename Tdata, typename... Args, size_t... Is>
    infiniStatus_t calculateImpl(const op::elementwise::ElementwiseInfo &info,
                                 void *output,
                                 const std::vector<const void *> &inputs,
                                 std::index_sequence<Is...>,
                                 cudaStream_t stream,
                                 Args &&...args) {
        if (info.output_size == 0) {
            return INFINI_STATUS_SUCCESS;
        }

        // casting the output and the inputs to Tdata pointers
        Tdata *out = reinterpret_cast<Tdata *>(output);
        const Tdata *inputs_arr[N];
        const Tdata **d_inputs_arr = nullptr;
        for (size_t i = 0; i < N; ++i) {
            inputs_arr[i] = reinterpret_cast<const Tdata *>(inputs[i]);
        }
        CHECK_CUDA(cudaMallocAsync(&d_inputs_arr, N * sizeof(*d_inputs_arr), stream));
        CHECK_CUDA(cudaMemcpyAsync(d_inputs_arr, inputs_arr, N * sizeof(*d_inputs_arr), cudaMemcpyHostToDevice, stream));

        // create and send the info to device
        const bool *d_bools = nullptr;
        const bool *d_input_contiguous = nullptr;
        const bool *d_input_broadcasted = nullptr;
        const int8_t *d_output_shape_strides = nullptr;
        const size_t *d_output_shape = nullptr;
        const ptrdiff_t *d_output_strides = nullptr;
        const size_t **d_input_shapes = nullptr;
        const ptrdiff_t **d_input_strides = nullptr;
        std::vector<const size_t *> tmp_device_ptrs(info.input_size);
        std::vector<const ptrdiff_t *> tmp_device_ptrs_strides(info.input_size);

        CHECK_STATUS(infoToDevice<N>(info, d_bools, d_input_contiguous,
                                     d_input_broadcasted, d_output_shape_strides, d_output_shape,
                                     d_output_strides, tmp_device_ptrs, d_input_shapes, tmp_device_ptrs_strides,
                                     d_input_strides, stream));

        dim3 blockDims(std::min(BLOCK_SIZE, static_cast<size_t>(internal->maxThreadsPerBlock())));
        dim3 gridDims(std::min(CEIL_DIV(info.output_size, blockDims.x), static_cast<size_t>(internal->gridSizeX())));
        size_t step = gridDims.x * blockDims.x;

        for (size_t i = 0; i < info.output_size; i += step) {
            elementwise_kernel<N, Op, Tdata, Args...><<<gridDims, blockDims, 0, stream>>>(
                info.output_size,
                info.ndim,
                info.output_contiguous,
                d_input_contiguous,
                d_input_broadcasted,
                d_output_shape,
                d_input_shapes,
                d_output_strides,
                d_input_strides,
                info.input_size, out, d_inputs_arr, i, std::forward<Args>(args)...);
        }

        CHECK_STATUS(freeAllDevice((const void **)d_inputs_arr, d_bools, d_output_shape_strides,
                                   info.input_size, d_input_shapes, d_input_strides, stream));
        return INFINI_STATUS_SUCCESS;
    }

    /**
     * @brief Performs elementwise operations when inputs and the outputs have mixed data types (i.e., different dtypes).
     *
     * @tparam BLOCK_SIZE  The block size for the kernel launch.
     * @tparam N           The number of input tensors.
     * @tparam Op          The operation to perform (e.g., addition, multiplication).
     * @tparam Tout        The output data type.
     * @tparam Tin         The input data types.
     * @tparam Args        Additional arguments to be passed to the operation.
     * @param info         Structure containing elementwise operation information (size, shape, etc.).
     * @param output       Pointer to the output memory where results will be stored.
     * @param inputs       Vector of pointers to input tensors.
     * @param stream       CUDA stream used for asynchronous execution.
     * @param args         Additional arguments for the operation.
     * @return infiniStatus_t  Status indicating success or failure.
     */
    template <size_t BLOCK_SIZE, size_t N, typename Op, typename Tout, typename... Tin, typename... Args, size_t... Is,
              std::enable_if_t<(sizeof...(Tin) == Op::num_inputs), int> = 0>
    infiniStatus_t calculateImpl(const op::elementwise::ElementwiseInfo &info,
                                 void *output,
                                 const std::vector<const void *> &inputs,
                                 std::index_sequence<Is...>,
                                 cudaStream_t stream,
                                 Args &&...args) {
        if (info.output_size == 0) {
            return INFINI_STATUS_SUCCESS;
        }

        Tout *out = reinterpret_cast<Tout *>(output);

        // Store input pointers with the correct types
        const std::tuple<const Tin *...> inputs_arr{reinterpret_cast<const Tin *>(inputs[Is])...};
        const void **d_inputs_arr = nullptr;

        // Create array of input pointers on host (void*) to copy to device
        const void *host_input_ptrs[] = {reinterpret_cast<const void *>(std::get<Is>(inputs_arr))...};
        CHECK_CUDA(cudaMallocAsync(&d_inputs_arr, N * sizeof(void *), stream));
        CHECK_CUDA(cudaMemcpyAsync(d_inputs_arr, host_input_ptrs, N * sizeof(void *), cudaMemcpyHostToDevice, stream));

        // Device pointers
        const bool *d_bools = nullptr;
        const bool *d_input_contiguous = nullptr;
        const bool *d_input_broadcasted = nullptr;
        const int8_t *d_output_shape_strides = nullptr;
        const size_t *d_output_shape = nullptr;
        const ptrdiff_t *d_output_strides = nullptr;
        const size_t **d_input_shapes = nullptr;
        const ptrdiff_t **d_input_strides = nullptr;
        std::vector<const size_t *> tmp_device_ptrs(info.input_size);
        std::vector<const ptrdiff_t *> tmp_device_ptrs_strides(info.input_size);

        CHECK_STATUS(infoToDevice<N>(info, d_bools, d_input_contiguous,
                                     d_input_broadcasted, d_output_shape_strides, d_output_shape,
                                     d_output_strides, tmp_device_ptrs, d_input_shapes, tmp_device_ptrs_strides,
                                     d_input_strides, stream));

        dim3 blockDims(std::min(BLOCK_SIZE, static_cast<size_t>(internal->maxThreadsPerBlock())));
        dim3 gridDims(std::min(CEIL_DIV(info.output_size, blockDims.x), static_cast<size_t>(internal->gridSizeX())));
        size_t step = gridDims.x * blockDims.x;

        for (size_t i = 0; i < info.output_size; i += step) {
            elementwise_kernel<Op, Tout, Tin...><<<gridDims, blockDims, 0, stream>>>(
                info.output_size,
                info.ndim,
                info.output_contiguous,
                d_input_contiguous,
                d_input_broadcasted,
                d_output_shape,
                d_input_shapes,
                d_output_strides,
                d_input_strides,
                info.input_size, out, reinterpret_cast<const void **>(d_inputs_arr), i);
        }

        CHECK_STATUS(freeAllDevice(d_inputs_arr, d_bools, d_output_shape_strides, info.input_size, d_input_shapes, d_input_strides, stream));
        return INFINI_STATUS_SUCCESS;
    }

private:
    /**
     * @brief Transfers elementwise kernel metadata (shapes, strides, flags) from host to device.
     *
     * @tparam N                      Number of inputs.
     * @param info                    Structure containing input/output metadata.
     * @param d_bools                 Device pointer for input_contiguous and input_broadcasted flags.
     * @param d_input_contiguous      Device pointer to input contiguity flags.
     * @param d_input_broadcasted     Device pointer to input broadcasting flags.
     * @param d_output_shape_strides  Device buffer containing both output shape and strides.
     * @param d_output_shape          Device pointer to output shape.
     * @param d_output_strides        Device pointer to output strides.
     * @param tmp_device_ptrs         Temporary device pointers for input shapes.
     * @param d_input_shapes          Device array of pointers to input shapes.
     * @param tmp_device_ptrs_strides Temporary device pointers for input strides.
     * @param d_input_strides         Device array of pointers to input strides.
     * @param stream                  CUDA stream for async allocation and transfers.
     * @return infiniStatus_t         Status indicating success or failure.
     */
    template <size_t N>
    infiniStatus_t infoToDevice(
        const op::elementwise::ElementwiseInfo &info,
        const bool *&d_bools,
        const bool *&d_input_contiguous,
        const bool *&d_input_broadcasted,
        const int8_t *&d_output_shape_strides,
        const size_t *&d_output_shape,
        const ptrdiff_t *&d_output_strides,
        std::vector<const size_t *> &tmp_device_ptrs,
        const size_t **&d_input_shapes,
        std::vector<const ptrdiff_t *> &tmp_device_ptrs_strides,
        const ptrdiff_t **&d_input_strides,
        cudaStream_t stream) const {

        CHECK_CUDA(cudaMallocAsync(&d_bools, 2 * info.input_size * sizeof(*d_bools), stream));
        CHECK_CUDA(cudaMemcpyAsync((void *)d_bools, info.input_contiguous, info.input_size * sizeof(*d_bools), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync((void *)(d_bools + info.input_size), info.input_broadcasted, info.input_size * sizeof(*d_bools), cudaMemcpyHostToDevice, stream));

        CHECK_CUDA(cudaMallocAsync(&d_output_shape_strides, info.ndim * (sizeof(*d_output_shape) + sizeof(*d_output_strides)), stream));
        CHECK_CUDA(cudaMemcpyAsync((void *)d_output_shape_strides, info.output_shape, info.ndim * sizeof(*d_output_shape), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync((void *)(d_output_shape_strides + info.ndim * sizeof(*d_output_shape)), info.output_strides, info.ndim * sizeof(*d_output_strides), cudaMemcpyHostToDevice, stream));

        CHECK_CUDA(cudaMallocAsync(&d_input_shapes, info.input_size * sizeof(*d_input_shapes), stream));
        for (size_t i = 0; i < info.input_size; ++i) {
            CHECK_CUDA(cudaMallocAsync(&tmp_device_ptrs[i], info.ndim * sizeof(*&tmp_device_ptrs[i]), stream));
            CHECK_CUDA(cudaMemcpyAsync((void *)tmp_device_ptrs[i], info.input_shapes[i],
                                       info.ndim * sizeof(*tmp_device_ptrs[i]), cudaMemcpyHostToDevice, stream));
        }
        CHECK_CUDA(cudaMemcpyAsync((void *)d_input_shapes, tmp_device_ptrs.data(),
                                   info.input_size * sizeof(*d_input_shapes), cudaMemcpyHostToDevice, stream));

        CHECK_CUDA(cudaMallocAsync(&d_input_strides, info.input_size * sizeof(*d_input_strides), stream));
        for (size_t i = 0; i < info.input_size; ++i) {
            CHECK_CUDA(cudaMallocAsync(&tmp_device_ptrs_strides[i], info.ndim * sizeof(*tmp_device_ptrs_strides[i]), stream));
            CHECK_CUDA(cudaMemcpyAsync((void *)tmp_device_ptrs_strides[i], info.input_strides[i],
                                       info.ndim * sizeof(*tmp_device_ptrs_strides[i]), cudaMemcpyHostToDevice, stream));
        }
        CHECK_CUDA(cudaMemcpyAsync((void *)d_input_strides, tmp_device_ptrs_strides.data(),
                                   info.input_size * sizeof(*d_input_strides), cudaMemcpyHostToDevice, stream));

        d_input_contiguous = d_bools;
        d_input_broadcasted = d_bools + info.input_size;
        d_output_shape = reinterpret_cast<const size_t *>(d_output_shape_strides);
        d_output_strides = reinterpret_cast<const ptrdiff_t *>(d_output_shape_strides + info.ndim * sizeof(*d_output_shape));

        return INFINI_STATUS_SUCCESS;
    }

    /**
     * @brief Frees all device-allocated memory used for metadata in elementwise kernel execution.
     *
     * @param d_inputs_arr            Device array of input pointers.
     * @param d_bools                 Device memory holding input flags.
     * @param d_output_shape_strides  Device buffer holding output shape and strides.
     * @param input_size              Number of input tensors.
     * @param d_input_shapes          Device array of input shape pointers.
     * @param d_input_strides         Device array of input stride pointers.
     * @param stream                  CUDA stream for async deallocation.
     * @return infiniStatus_t         Status indicating success or failure.
     */
    inline infiniStatus_t freeAllDevice(const void **d_inputs_arr,
                                        const bool *d_bools,
                                        const int8_t *d_output_shape_strides,
                                        const size_t input_size,
                                        const size_t **d_input_shapes,
                                        const ptrdiff_t **d_input_strides,
                                        cudaStream_t stream) const {

        CHECK_CUDA(cudaFreeAsync((void *)d_inputs_arr, stream));
        CHECK_CUDA(cudaFreeAsync((void *)d_bools, stream));
        CHECK_CUDA(cudaFreeAsync((void *)d_output_shape_strides, stream));
        CHECK_CUDA(cudaFreeAsync((void *)d_input_shapes, stream));
        CHECK_CUDA(cudaFreeAsync((void *)d_input_strides, stream));
        return INFINI_STATUS_SUCCESS;
    }
};

template <typename... Args>
infiniStatus_t DeviceImpl::create(DeviceImpl **device_info,
                                  Args &&...args) {
    auto opaque = std::make_shared<Opaque>(std::forward<Args>(args)...);
    *device_info = new DeviceImpl(opaque);
    return INFINI_STATUS_SUCCESS;
}

/* Invoke elementwise operation for different input types */
template <unsigned int BLOCK_SIZE, typename Op, typename Tout, typename... Tin, typename... Args,
          std::enable_if_t<(sizeof...(Tin) == Op::num_inputs), int>>
infiniStatus_t DeviceImpl::calculate(const op::elementwise::ElementwiseInfo &info,
                                     void *output,
                                     const std::vector<const void *> &inputs,
                                     void *stream,
                                     Args &&...args) {
    constexpr size_t N = Op::num_inputs;
    static_assert(sizeof...(Tin) == N, "Input type count mismatch");
    return _opaque->calculateImpl<BLOCK_SIZE, N, Op, Tout, Tin...>(
        info, output, inputs,
        std::make_index_sequence<N>{},
        reinterpret_cast<cudaStream_t>(stream),
        std::forward<Args>(args)...);
}

/* Invoke elementwise operation when all inputs have the same dtype */
template <unsigned int BLOCK_SIZE, typename Op, typename Tdata, typename... Args>
infiniStatus_t DeviceImpl::calculate(const op::elementwise::ElementwiseInfo &info,
                                     void *output,
                                     const std::vector<const void *> &inputs,
                                     void *stream,
                                     Args &&...args) {
    constexpr size_t N = Op::num_inputs;
    return _opaque->calculateImpl<BLOCK_SIZE, N, Op, Tdata>(
        info, output, inputs,
        std::make_index_sequence<N>{},
        reinterpret_cast<cudaStream_t>(stream),
        std::forward<Args>(args)...);
}

} // namespace op::elementwise::cuda

#endif // __INFINIOP_ELEMENTWISE_CUDA_H__
