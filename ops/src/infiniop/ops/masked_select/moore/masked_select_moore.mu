#include "../../../devices/moore/moore_common.h"
#include "masked_select_moore.h"

#include "../../../devices/moore/moore_kernel_common.h"

// #include "infinicore/context/context.hpp"
#include "masked_select_moore_kernel.h"

namespace op::masked_select::moore {

struct Descriptor::Opaque {
    std::shared_ptr<device::moore::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t mask_desc) {

    auto result = MaskedSelectInfo::create(input_desc, mask_desc);
    CHECK_RESULT(result);
    auto info = result.take();
    size_t workspace_size = 0;
    // shape size
    workspace_size += info.ndim * sizeof(size_t);
    // strides size * 2
    workspace_size += info.ndim * sizeof(ptrdiff_t) * 2;
    // size_t mark_scan
    workspace_size += info.total_elements * sizeof(size_t);
    // size_t mark_result
    workspace_size += info.total_elements * sizeof(size_t);
    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::moore::Handle *>(handle)->internal()},
        info, workspace_size, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

namespace {

template <size_t BLOCK_SIZE, typename T>
infiniStatus_t launchKernel(
    const MaskedSelectInfo &info,
    const T *input, const bool *mask, void **data_ptr, size_t *dlen_ptr,
    musaStream_t stream, void *workspace, size_t workspace_size) {

    size_t ndim = info.ndim;
    size_t total_elements = info.total_elements;

    size_t workspace_offset = 0;
    unsigned char *workspace_ptr = reinterpret_cast<unsigned char *>(workspace);

    size_t *shape_musa = reinterpret_cast<size_t *>(workspace_ptr + workspace_offset);
    workspace_offset += ndim * sizeof(size_t);

    ptrdiff_t *input_strides_musa = reinterpret_cast<ptrdiff_t *>(workspace_ptr + workspace_offset);
    workspace_offset += ndim * sizeof(ptrdiff_t);

    ptrdiff_t *mask_strides_musa = reinterpret_cast<ptrdiff_t *>(workspace_ptr + workspace_offset);
    workspace_offset += ndim * sizeof(ptrdiff_t);

    size_t *mark_scan = reinterpret_cast<size_t *>(workspace_ptr + workspace_offset);
    workspace_offset += info.total_elements * sizeof(size_t);

    size_t *scan_result = reinterpret_cast<size_t *>(workspace_ptr + workspace_offset);
    workspace_offset += info.total_elements * sizeof(size_t);

    assert(workspace_offset == workspace_size);

    CHECK_MOORE(musaMemcpyAsync(shape_musa, info.shape.data(), ndim * sizeof(size_t), musaMemcpyHostToDevice, stream));
    CHECK_MOORE(musaMemcpyAsync(input_strides_musa, info.input_strides.data(), ndim * sizeof(ptrdiff_t), musaMemcpyHostToDevice, stream));
    CHECK_MOORE(musaMemcpyAsync(mask_strides_musa, info.mask_strides.data(), ndim * sizeof(ptrdiff_t), musaMemcpyHostToDevice, stream));

    size_t block_size = BLOCK_SIZE;
    size_t grid_size = (total_elements + block_size - 1) / block_size;

    maskedSelectGetMarkScanOnceKernel<BLOCK_SIZE><<<grid_size, block_size, 0, stream>>>(
        mask, mark_scan, total_elements,
        shape_musa, mask_strides_musa, ndim);
    CHECK_MOORE(musaDeviceSynchronize());

    for (size_t stride = BLOCK_SIZE; stride < total_elements; stride *= BLOCK_SIZE) {
        size_t stride_elements = (total_elements + stride - 1) / stride;
        size_t stride_grid_size = (stride_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

        maskedSelectScanWithStrideKernel<BLOCK_SIZE><<<stride_grid_size, block_size, 0, stream>>>(
            mark_scan, total_elements, stride);
        CHECK_MOORE(musaDeviceSynchronize());
    }
    if (total_elements > BLOCK_SIZE) {
        maskedSelectCountScanResultKernel<BLOCK_SIZE><<<grid_size, block_size, 0, stream>>>(
            mark_scan, scan_result, total_elements);
        CHECK_MOORE(musaDeviceSynchronize());
    } else {
        scan_result = mark_scan;
    }

    CHECK_MOORE(musaMemcpyAsync(dlen_ptr, scan_result + total_elements - 1, sizeof(size_t), musaMemcpyDeviceToHost, stream));

    CHECK_MOORE(musaMalloc(data_ptr, *dlen_ptr * sizeof(T)));

    // context::setDevice(MOORE);
    // context::allocateMemory(data_ptr, *dlen_ptr * sizeof(T));

    maskedSelectGetDataKernel<BLOCK_SIZE, T><<<grid_size, block_size, 0, stream>>>(
        input, mask, scan_result, (T *)*data_ptr, total_elements,
        shape_musa, input_strides_musa, mask_strides_musa, ndim);
    CHECK_MOORE(musaDeviceSynchronize());

    return INFINI_STATUS_SUCCESS;
}

} // namespace

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    const void *input,
    const bool *mask,
    void **data_ptr,
    size_t *dlen_ptr,
    void *stream_) const {

    musaStream_t stream = (musaStream_t)stream_;
#define CALCULATE_MASKED_SELECT(BLOCK_SIZE, T)      \
    launchKernel<BLOCK_SIZE, T>(                    \
        _info,                                      \
        (const T *)input, mask, data_ptr, dlen_ptr, \
        stream, workspace, workspace_size)
#define CALCULATE_MASKED_SELECT_WITH_BLOCK_SIZE(BLOCK_SIZE)    \
    {                                                          \
        if (_info.dtype == INFINI_DTYPE_F32)                   \
            return CALCULATE_MASKED_SELECT(BLOCK_SIZE, float); \
        else                                                   \
            return INFINI_STATUS_BAD_TENSOR_DTYPE;             \
    }

    if (_opaque->internal->maxThreadsPerBlock() == MOORE_BLOCK_SIZE_1024) {
        CALCULATE_MASKED_SELECT_WITH_BLOCK_SIZE(MOORE_BLOCK_SIZE_1024);
    } else if (_opaque->internal->maxThreadsPerBlock() == MOORE_BLOCK_SIZE_512) {
        CALCULATE_MASKED_SELECT_WITH_BLOCK_SIZE(MOORE_BLOCK_SIZE_512);
    } else if (_opaque->internal->maxThreadsPerBlock() == MOORE_BLOCK_SIZE_2048) {
        CALCULATE_MASKED_SELECT_WITH_BLOCK_SIZE(MOORE_BLOCK_SIZE_2048);
    } else {
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    }
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::masked_select::moore
