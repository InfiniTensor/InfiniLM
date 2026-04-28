#include "../../../devices/moore/moore_common.h"
#include "inner_moore.h"

#include "../../../devices/moore/moore_kernel_common.h"

#include "inner_moore_kernel.h"

namespace op::inner::moore {

struct Descriptor::Opaque {
    std::shared_ptr<device::moore::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t other_desc) {

    auto result = InnerInfo::create(out_desc, input_desc, other_desc);
    CHECK_RESULT(result);
    auto info = result.take();
    size_t workspace_size = 0;
    // out_shape
    workspace_size += info.out_ndim * sizeof(size_t);
    // input, other, out strides
    workspace_size += (info.input_ndim + info.other_ndim + info.out_ndim) * sizeof(ptrdiff_t);
    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::moore::Handle *>(handle)->internal()},
        info, workspace_size, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

namespace {

template <size_t BLOCK_SIZE, typename T>
infiniStatus_t launchKernel(
    const InnerInfo &info,
    const T *input, const T *other, T *out,
    musaStream_t stream, void *workspace, size_t workspace_size) {

    size_t input_ndim = info.input_ndim;
    size_t other_ndim = info.other_ndim;
    size_t out_ndim = info.out_ndim;
    size_t total_elements = info.total_elements;
    size_t oper_len = info.oper_len;

    size_t workspace_offset = 0;
    unsigned char *workspace_ptr = reinterpret_cast<unsigned char *>(workspace);

    size_t *out_shape_musa = reinterpret_cast<size_t *>(workspace_ptr + workspace_offset);
    workspace_offset += out_ndim * sizeof(size_t);

    ptrdiff_t *input_strides_musa = reinterpret_cast<ptrdiff_t *>(workspace_ptr + workspace_offset);
    ptrdiff_t *other_strides_musa = input_strides_musa + input_ndim;
    ptrdiff_t *out_strides_musa = other_strides_musa + other_ndim;
    workspace_offset += (info.input_ndim + info.other_ndim + info.out_ndim) * sizeof(ptrdiff_t);

    assert(workspace_offset == workspace_size);

    CHECK_MOORE(musaMemcpyAsync(out_shape_musa, info.out_shape.data(), out_ndim * sizeof(size_t), musaMemcpyHostToDevice, stream));
    CHECK_MOORE(musaMemcpyAsync(input_strides_musa, info.input_strides.data(), input_ndim * sizeof(ptrdiff_t), musaMemcpyHostToDevice, stream));
    CHECK_MOORE(musaMemcpyAsync(other_strides_musa, info.other_strides.data(), other_ndim * sizeof(ptrdiff_t), musaMemcpyHostToDevice, stream));
    CHECK_MOORE(musaMemcpyAsync(out_strides_musa, info.out_strides.data(), out_ndim * sizeof(ptrdiff_t), musaMemcpyHostToDevice, stream));

    size_t block_size = BLOCK_SIZE;
    size_t grid_size = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    innerKernel<BLOCK_SIZE, T><<<grid_size, block_size, 0, stream>>>(
        input, other, out,
        total_elements, oper_len, out_shape_musa,
        input_strides_musa, other_strides_musa, out_strides_musa,
        input_ndim, other_ndim, out_ndim);

    return INFINI_STATUS_SUCCESS;
}

} // namespace

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *out,
    const void *input,
    const void *other,
    void *stream_) const {

    musaStream_t stream = (musaStream_t)stream_;
#define CALCULATE_INNER(BLOCK_SIZE, T)                \
    launchKernel<BLOCK_SIZE, T>(                      \
        _info,                                        \
        (const T *)input, (const T *)other, (T *)out, \
        stream, workspace, workspace_size)
#define CALCULATE_INNER_WITH_BLOCK_SIZE(BLOCK_SIZE)            \
    {                                                          \
        if (_info.dtype == INFINI_DTYPE_BF16)                  \
            return CALCULATE_INNER(BLOCK_SIZE, __mt_bfloat16); \
        else if (_info.dtype == INFINI_DTYPE_F16)              \
            return CALCULATE_INNER(BLOCK_SIZE, half);          \
        else if (_info.dtype == INFINI_DTYPE_F32)              \
            return CALCULATE_INNER(BLOCK_SIZE, float);         \
        else                                                   \
            return INFINI_STATUS_BAD_TENSOR_DTYPE;             \
    }

    if (_opaque->internal->maxThreadsPerBlock() == MOORE_BLOCK_SIZE_1024) {
        CALCULATE_INNER_WITH_BLOCK_SIZE(MOORE_BLOCK_SIZE_1024)
    } else if (_opaque->internal->maxThreadsPerBlock() == MOORE_BLOCK_SIZE_512) {
        CALCULATE_INNER_WITH_BLOCK_SIZE(MOORE_BLOCK_SIZE_512)
    } else if (_opaque->internal->maxThreadsPerBlock() == MOORE_BLOCK_SIZE_2048) {
        CALCULATE_INNER_WITH_BLOCK_SIZE(MOORE_BLOCK_SIZE_2048)
    } else {
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    }
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::inner::moore
