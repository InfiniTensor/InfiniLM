#ifndef __INFINIOP_ELEMENTWISE_KUNLUN_H__
#define __INFINIOP_ELEMENTWISE_KUNLUN_H__

#include "../../../utils.h"
#include "../../devices/kunlun/kunlun_handle.h"
#include "elementwise_kunlun_api.h"

namespace op::elementwise::kunlun {

struct DeviceImpl::Opaque {
    std::shared_ptr<device::kunlun::Handle::Internal> internal;

    Opaque(const std::shared_ptr<device::kunlun::Handle::Internal> &internal_)
        : internal(internal_) {}

    template <size_t N, typename Op, typename Tdata, typename... Args>
    infiniStatus_t calculateImpl(const op::elementwise::ElementwiseInfo &info,
                                 void *workspace,
                                 void *output,
                                 const std::vector<const void *> &inputs,
                                 kunlunStream_t stream,
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
                                     d_input_shapes, d_input_strides));

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
            stream,
            args...);

        return INFINI_STATUS_SUCCESS;
    }

private:
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

        // copy the input pointer array and meta to device
        CHECK_KUNLUN(xpu_memcpy(workspace, h_inputs_arr, input_arr_size, XPU_HOST_TO_DEVICE));
        CHECK_KUNLUN(xpu_memcpy((void *)d_meta_start, info_meta_start, info.getMetaMemSize(), XPU_HOST_TO_DEVICE));

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
};

template <typename... Args>
utils::Result<DeviceImpl *> DeviceImpl::create(Args &&...args) {
    auto opaque = std::make_shared<Opaque>(std::forward<Args>(args)...);
    return utils::Result<DeviceImpl *>(new DeviceImpl(opaque));
}

template <typename Op, typename Tdata, typename... Args>
infiniStatus_t DeviceImpl::calculate(const op::elementwise::ElementwiseInfo &info,
                                     void *workspace,
                                     void *output,
                                     const std::vector<const void *> &inputs,
                                     void *stream,
                                     Args &&...args) {
    constexpr size_t N = Op::num_inputs;
    return _opaque->calculateImpl<N, Op, Tdata>(
        info, workspace, output, inputs,
        reinterpret_cast<kunlunStream_t>(stream),
        std::forward<Args>(args)...);
}
} // namespace op::elementwise::kunlun

// Template for kunlun kernel interface declaration
#define LAUNCH_ELEMENTWISE_KERNEL(OpName)       \
    template <typename Tdata, typename... Args> \
    void launch##OpName##Kernel(                \
        size_t output_size,                     \
        size_t ndim,                            \
        bool output_contiguous,                 \
        const void *input_contiguous,           \
        const void *input_broadcasted,          \
        const void *output_shape,               \
        const void *input_shapes,               \
        const void *output_strides,             \
        const void *input_strides,              \
        void *output,                           \
        const void *const *inputs,              \
        XPUStream stream,                       \
        Args... args);

#endif
