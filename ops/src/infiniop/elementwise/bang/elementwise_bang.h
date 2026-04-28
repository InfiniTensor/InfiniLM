#ifndef __INFINIOP_ELEMENTWISE_BANG_H__
#define __INFINIOP_ELEMENTWISE_BANG_H__

#include "../../../utils.h"
#include "../../devices/bang/common_bang.h"
#include "elementwise_bang_api.h"

namespace op::elementwise::bang {

/**
 * @brief Opaque implementation structure for BANG device operations.
 *
 * Contains device-specific resources and implementation methods.
 */
struct DeviceImpl::Opaque {
    std::shared_ptr<device::bang::Handle::Internal> internal;

    /**
     * @brief Constructs an Opaque instance with device handle internals.
     *
     * @param internal_ Shared pointer to BANG device handle internals.
     */
    Opaque(const std::shared_ptr<device::bang::Handle::Internal> &internal_)
        : internal(internal_) {}

    /**
     * @brief Implements elementwise calculation for BANG device.
     *
     * @tparam N        Number of input tensors.
     * @tparam Op       Operator functor type.
     * @tparam Tdata    Data type for inputs and output.
     * @tparam Args     Additional arguments for the operator.
     *
     * @param info      Elementwise operation metadata (shapes, strides, etc.).
     * @param workspace Device workspace memory.
     * @param output    Output tensor buffer.
     * @param inputs    Vector of input tensor pointers.
     * @param queue     BANG queue for asynchronous execution.
     * @param args      Additional arguments for the operator.
     * @return infiniStatus_t Status indicating success or failure.
     */
    template <size_t N, typename Op, typename Tdata, typename... Args>
    infiniStatus_t calculateImpl(const op::elementwise::ElementwiseInfo &info,
                                 void *workspace,
                                 void *output,
                                 const std::vector<const void *> &inputs,
                                 cnrtQueue_t queue,
                                 Args &&...args) {
        auto output_size = info.getOutputSize();
        if (output_size == 0) {
            return INFINI_STATUS_SUCCESS;
        }

        // Device pointers for metadata
        const void **d_inputs_arr = nullptr;
        const bool *d_input_contiguous = nullptr;
        const bool *d_input_broadcasted = nullptr;
        const size_t *d_output_shape = nullptr;
        const ptrdiff_t *d_output_strides = nullptr;
        const size_t *d_input_shapes = nullptr;
        const ptrdiff_t *d_input_strides = nullptr;

        // Copy metadata to device and setup pointers
        CHECK_STATUS(infoToDevice<N>(info, workspace, inputs.data(), d_inputs_arr,
                                     d_input_contiguous, d_input_broadcasted,
                                     d_output_shape, d_output_strides,
                                     d_input_shapes, d_input_strides));

        // Launch the elementwise kernel
        Op::template launch<Tdata>(
            output_size,
            info.getNdim(),
            info.isOutputContiguous(),
            reinterpret_cast<const void *>(d_input_contiguous),
            reinterpret_cast<const void *>(d_input_broadcasted),
            reinterpret_cast<const void *>(d_output_shape),
            reinterpret_cast<const void *>(d_input_shapes),
            reinterpret_cast<const void *>(d_output_strides),
            reinterpret_cast<const void *>(d_input_strides),
            output,
            reinterpret_cast<const void *const *>(d_inputs_arr),
            queue,
            internal,
            args...);

        // Synchronize queue to ensure completion
        CNRT_CHECK(cnrtQueueSync(queue));

        return INFINI_STATUS_SUCCESS;
    }

private:
    /**
     * @brief Transfers elementwise operation metadata to device memory.
     *
     * @tparam N                     Number of input tensors.
     *
     * @param info                   Elementwise operation metadata.
     * @param workspace              Device workspace memory.
     * @param h_inputs_arr           Host array of input pointers.
     * @param d_inputs_arr           Output reference to device input pointers.
     * @param d_input_contiguous     Output reference to contiguous flags.
     * @param d_input_broadcasted    Output reference to broadcast flags.
     * @param d_output_shape         Output reference to output shape.
     * @param d_output_strides       Output reference to output strides.
     * @param d_input_shapes         Output reference to input shapes.
     * @param d_input_strides        Output reference to input strides.
     * @return infiniStatus_t        Status indicating success or failure.
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
        const ptrdiff_t *&d_input_strides) const {

        constexpr auto input_size = N;
        const auto ndim = info.getNdim();
        constexpr auto input_arr_size = N * sizeof(*h_inputs_arr);
        const int8_t *info_meta_start = info.getMetaStart();
        const int8_t *d_meta_start = reinterpret_cast<int8_t *>(workspace) + input_arr_size;

        // Copy input pointer array and metadata to device
        CNRT_CHECK(cnrtMemcpy(workspace, (void *)h_inputs_arr, input_arr_size, cnrtMemcpyHostToDev));
        CNRT_CHECK(cnrtMemcpy((void *)d_meta_start, (void *)info_meta_start, info.getMetaMemSize(), cnrtMemcpyHostToDev));

        // Setup pointers to device memory regions
        d_inputs_arr = reinterpret_cast<const void **>(workspace);
        d_output_shape = reinterpret_cast<const size_t *>(d_meta_start);
        d_output_strides = reinterpret_cast<const ptrdiff_t *>(d_output_shape + ndim);
        d_input_shapes = reinterpret_cast<const size_t *>(d_output_strides + ndim);
        d_input_strides = reinterpret_cast<const ptrdiff_t *>(d_input_shapes + input_size * ndim);
        d_input_contiguous = reinterpret_cast<const bool *>(d_input_strides + input_size * ndim);
        d_input_broadcasted = reinterpret_cast<const bool *>(d_input_contiguous + input_size);

        return INFINI_STATUS_SUCCESS;
    }
};

/**
 * @brief Creates a DeviceImpl instance for BANG device.
 *
 * @tparam Args Argument types for Opaque construction.
 * @param args Arguments forwarded to Opaque constructor.
 * @return utils::Result<DeviceImpl*> Result containing new DeviceImpl instance.
 */
template <typename... Args>
utils::Result<DeviceImpl *> DeviceImpl::create(Args &&...args) {
    auto opaque = std::make_shared<Opaque>(std::forward<Args>(args)...);
    return utils::Result<DeviceImpl *>(new DeviceImpl(opaque));
}

/**
 * @brief Calculates elementwise operation for BANG device.
 *
 * @tparam Op       Operator functor type.
 * @tparam Tdata    Data type for inputs and output.
 * @tparam Args     Additional arguments for the operator.
 *
 * @param info      Elementwise operation metadata.
 * @param workspace Device workspace memory.
 * @param output    Output tensor buffer.
 * @param inputs    Vector of input tensor pointers.
 * @param queue     BANG queue (as void*).
 * @param args      Additional arguments for the operator.
 * @return infiniStatus_t Status indicating success or failure.
 */
template <typename Op, typename Tdata, typename... Args>
infiniStatus_t DeviceImpl::calculate(const op::elementwise::ElementwiseInfo &info,
                                     void *workspace,
                                     void *output,
                                     const std::vector<const void *> &inputs,
                                     void *queue,
                                     Args &&...args) {
    constexpr size_t N = Op::num_inputs;
    return _opaque->calculateImpl<N, Op, Tdata>(
        info, workspace, output, inputs,
        reinterpret_cast<cnrtQueue_t>(queue),
        std::forward<Args>(args)...);
}
} // namespace op::elementwise::bang

/**
 * @brief Macro for declaring BANG kernel interface.
 *
 * @param OpName Name of the elementwise operation.
 */
#define LAUNCH_ELEMENTWISE_KERNEL(OpName)                                \
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
        cnrtQueue_t queue,                                               \
        const std::shared_ptr<device::bang::Handle::Internal> &internal, \
        Args... args);

#endif // __INFINIOP_ELEMENTWISE_BANG_H__
