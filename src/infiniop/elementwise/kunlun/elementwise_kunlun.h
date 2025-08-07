#ifndef __INFINIOP_ELEMENTWISE_KUNLUN_XPU__
#define __INFINIOP_ELEMENTWISE_KUNLUN_XPU__

#include "../../../utils.h"
#include "../../devices/kunlun/kunlun_common.h"
#include "../../devices/kunlun/kunlun_kernel_common.h"
#include "elementwise_kunlun_api.h"

namespace op::elementwise::kunlun {

using namespace device::kunlun::kernel;

template <typename T>
__device__ const T *typedInputPtr(const void *ptr) {
    return reinterpret_cast<const T *>(ptr);
}

/**
 * @brief Computes input tile offset
 */
struct InputIndexer {
    int idx;
    int ndim;
    const bool *input_contiguous;
    const bool *input_broadcasted;
    const _size_t *input_shapes;
    const _ptrdiff_t *input_strides;
    const _ptrdiff_t *output_strides;

    inline __device__ int operator()(int input_id) const {
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
inline __device__ int
getOutputIndex(int idx,
               bool is_contiguous,
               int ndim,
               const _size_t *shape,
               const _ptrdiff_t *strides) {
    return is_contiguous ? idx : indexToOffset(idx, ndim, shape, strides);
}

/**
 * @brief Computes elements of input indexes
 */
template <int N, typename Op, typename Tdata, typename... Args>
__device__ void launchOp(
    __global_ptr__ Tdata **typed_inputs, // gm pointer
    __global_ptr__ Tdata *output,        // gm pointer output
    Tdata *inputs_buf,                   // local mem buffer
    int *input_indexes,
    int output_index,
    Args... args) {

    static_assert(N == Op::num_inputs, "template N is not equal to Op::num_inputs!\n");

#pragma unroll
    // Copy inputs to buf
    for (int i = 0; i < N; i++) {
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

template <int N, typename Op, typename Tdata, typename... Args>
__global__ void elementwiseKernel(
    int output_size,
    int ndim,
    bool output_contiguous,
    const bool *input_contiguous_gm,
    const bool *input_broadcasted_gm,
    const void *output_shape_gm,
    const void *input_shapes_gm,
    const void *output_strides_gm,
    const void *input_strides_gm,
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
        int read_len = min(len_per_loop, output_size - start);
        for (int idx = start; idx < start + read_len; ++idx) {
            int out_idx = getOutputIndex(idx, output_contiguous,
                                         ndim, output_shape, output_strides);
            InputIndexer indexer{idx, ndim, input_contiguous, input_broadcasted,
                                 input_shapes, input_strides, output_strides};
            // Get index offset for every operand
            int indexes[N];
            for (int i = 0; i < N; i++) {
                indexes[i] = indexer(i);
            }
            // Launch operater
            launchOp<N, Op, Tdata>(&typed_inputs_ptr[0], output, inputs_buf, indexes, out_idx, args...);
        }
    }
    sync_cluster();
}

struct DeviceImpl::Opaque {
    std::shared_ptr<device::kunlun::Handle::Internal> internal;

    Opaque(const std::shared_ptr<device::kunlun::Handle::Internal> &internal_)
        : internal(internal_) {}

    template <uint32_t BLOCK_SIZE, int N, typename Op, typename Tdata, typename... Args>
    infiniStatus_t calculateImpl(const op::elementwise::ElementwiseInfo &info,
                                 void *workspace,
                                 void *output,
                                 const std::vector<const void *> &inputs,
                                 kunlunStream_t stream,
                                 Args &&...args) {
        return launchElementwiseKernel<BLOCK_SIZE, N>(
            info,
            workspace,
            reinterpret_cast<Tdata *>(output),
            inputs,
            elementwiseKernel<N, Op, Tdata, Args...>,
            stream,
            std::forward<Args>(args)...);
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
     * @param d_inputs_arr           Input reference to device array of input tensor pointers.
     * @param d_input_contiguous     Input reference to device array indicating whether each input is contiguous.
     * @param d_input_broadcasted    Input reference to device array indicating whether each input is broadcasted.
     * @param d_output_shape         Output reference to device array holding the output tensor shape.
     * @param d_output_strides       Output reference to device array holding output tensor strides.
     * @param d_input_shapes         Output reference to flattened input tensor shapes (N * ndim).
     * @param d_input_strides        Output reference to flattened input tensor strides (N * ndim).
     * @param stream                 KUNLUN stream used for asynchronous memory transfer.
     * @return infiniStatus_t        Status indicating success or failure of the memory transfer and setup.
     */
    template <int N>
    infiniStatus_t infoToDevice(
        const op::elementwise::ElementwiseInfo &info,
        void *workspace,
        const void *const *h_inputs_arr,
        __global_ptr__ const void **&d_inputs_arr,
        __global_ptr__ const bool *&d_input_contiguous,
        __global_ptr__ const bool *&d_input_broadcasted,
        __global_ptr__ const size_t *&d_output_shape,
        __global_ptr__ const ptrdiff_t *&d_output_strides,
        __global_ptr__ const size_t *&d_input_shapes,
        __global_ptr__ const ptrdiff_t *&d_input_strides,
        kunlunStream_t stream) const {

        constexpr auto input_size = N;
        const auto ndim = info.getNdim();
        constexpr auto input_arr_size = N * sizeof(*h_inputs_arr);
        auto info_meta_start = info.getMetaStart(); // host meta pointer

        auto d_meta_start = reinterpret_cast<__global_ptr__ int8_t *>(workspace)
                          + input_arr_size; // device meta pointer

        // copy the input pointer array and meta to device
        CHECK_KUNLUN(xpu_memcpy_async(workspace, h_inputs_arr, input_arr_size, XPU_HOST_TO_DEVICE, stream));
        CHECK_KUNLUN(xpu_memcpy_async((void *)d_meta_start, info_meta_start, info.getMetaMemSize(), XPU_HOST_TO_DEVICE, stream));

        xpu_wait(stream);
        // xpu_wait(stream);

        // offset/assign the pointers
        d_inputs_arr = reinterpret_cast<__global_ptr__ const void **>(workspace);
        d_output_shape = reinterpret_cast<__global_ptr__ const size_t *>(d_meta_start);
        d_output_strides = reinterpret_cast<__global_ptr__ const ptrdiff_t *>(d_output_shape + ndim);
        d_input_shapes = reinterpret_cast<__global_ptr__ const size_t *>(d_output_strides + ndim);
        d_input_strides = reinterpret_cast<__global_ptr__ const ptrdiff_t *>(d_input_shapes + input_size * ndim);
        d_input_contiguous = reinterpret_cast<const bool *>(d_input_strides + input_size * ndim);
        d_input_broadcasted = reinterpret_cast<const bool *>(d_input_contiguous + input_size);

        // contiguous / broadcast 信息
        const bool *contiguous = info.getInputContiguous();
        const bool *broadcasted = info.getInputBroadcasted();

        return INFINI_STATUS_SUCCESS;
    }

    /**
     * @brief Launch elementwise kernel
     */
    template <uint32_t BLOCK_SIZE, int N, typename KernelFunc, typename Tout, typename... Args>
    infiniStatus_t launchElementwiseKernel(
        const op::elementwise::ElementwiseInfo &info,
        void *workspace,
        Tout *output,
        const std::vector<const void *> &inputs,
        KernelFunc kernel_func,
        kunlunStream_t stream,
        Args &&...args) {

        auto output_size = info.getOutputSize();
        if (output_size == 0) {
            return INFINI_STATUS_SUCCESS;
        }

        // Device pointers
        __global_ptr__ const void **d_inputs_arr = nullptr;
        __global_ptr__ const bool *d_input_contiguous = nullptr;
        __global_ptr__ const bool *d_input_broadcasted = nullptr;
        __global_ptr__ const size_t *d_output_shape = nullptr;
        __global_ptr__ const ptrdiff_t *d_output_strides = nullptr;
        __global_ptr__ const size_t *d_input_shapes = nullptr;
        __global_ptr__ const ptrdiff_t *d_input_strides = nullptr;

        CHECK_STATUS(infoToDevice<N>(info, workspace, inputs.data(), d_inputs_arr,
                                     d_input_contiguous, d_input_broadcasted,
                                     d_output_shape, d_output_strides,
                                     d_input_shapes, d_input_strides, stream));

        kernel_func<<<BLOCK_SIZE, 64, stream>>>(
            output_size,
            info.getNdim(),
            info.isOutputContiguous(),
            d_input_contiguous,
            d_input_broadcasted,
            reinterpret_cast<__global_ptr__ const void *>(d_output_shape),
            reinterpret_cast<__global_ptr__ const void *>(d_input_shapes),
            reinterpret_cast<__global_ptr__ const void *>(d_output_strides),
            reinterpret_cast<__global_ptr__ const void *>(d_input_strides),
            output,
            reinterpret_cast<__global_ptr__ const void **>(d_inputs_arr),
            args...);

        return INFINI_STATUS_SUCCESS;
    }
};

template <typename... Args>
utils::Result<DeviceImpl *> DeviceImpl::create(Args &&...args) {
    auto opaque = std::make_shared<Opaque>(std::forward<Args>(args)...);
    return utils::Result<DeviceImpl *>(new DeviceImpl(opaque));
}

template <unsigned int BLOCK_SIZE, typename Op, typename Tdata, typename... Args>
infiniStatus_t DeviceImpl::calculate(const op::elementwise::ElementwiseInfo &info,
                                     void *workspace,
                                     void *output,
                                     const std::vector<const void *> &inputs,
                                     void *stream,
                                     Args &&...args) {
    constexpr int N = Op::num_inputs;
    return _opaque->calculateImpl<BLOCK_SIZE, N, Op, Tdata>(
        info, workspace, output, inputs,
        reinterpret_cast<kunlunStream_t>(stream),
        std::forward<Args>(args)...);
}

#define INSTANTIATE_ELEMENTWISE_KERNEL(N, Op, Tdata, ...)                    \
    template __global__ void elementwiseKernel<N, Op, Tdata, ##__VA_ARGS__>( \
        int output_size,                                                     \
        int ndim,                                                            \
        bool output_contiguous,                                              \
        const bool *input_contiguous_gm,                                     \
        const bool *input_broadcasted_gm,                                    \
        const void *output_shape_gm,                                         \
        const void *input_shapes_gm,                                         \
        const void *output_strides_gm,                                       \
        const void *input_strides_gm,                                        \
        Tdata *output,                                                       \
        const void *const *inputs,                                           \
        ##__VA_ARGS__);

} // namespace op::elementwise::kunlun

#endif
