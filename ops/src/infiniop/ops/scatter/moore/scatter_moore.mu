#include "../../../devices/moore/moore_handle.h"
#include "scatter_moore.h"
#include "scatter_moore_kernel.h"
#include <algorithm>
#include <cstdint>
#include <vector>

namespace op::scatter::moore {

// ==================================================================
// 1. Common Opaque Structure
// ==================================================================
struct ScatterMooreOpaque {
    op::scatter::moore::TensorGeometry geometry;
    size_t input_bytes;

    ScatterMooreOpaque(const infiniopTensorDescriptor_t updates_desc,
                       const infiniopTensorDescriptor_t indices_desc,
                       const infiniopTensorDescriptor_t output_desc) {

        geometry.ndim = static_cast<int>(updates_desc->ndim());

        // Calculate Input bytes for copy
        size_t total_elements = 1;
        for (size_t i = 0; i < output_desc->ndim(); ++i) {
            total_elements *= output_desc->shape()[i];
        }

        size_t dt_size = 0;
        if (output_desc->dtype() == INFINI_DTYPE_F32) {
            dt_size = 4;
        } else if (output_desc->dtype() == INFINI_DTYPE_F64) {
            dt_size = 8;
        } else {
            dt_size = 2; // f16/bf16
        }

        input_bytes = total_elements * dt_size;

        // Fill Geometry
        int ndim = geometry.ndim;
        for (int i = 0; i < ndim; ++i) {
            geometry.updates_shape[i] = updates_desc->shape()[i];
            geometry.updates_strides[i] = updates_desc->strides()[i];
            geometry.output_strides[i] = output_desc->strides()[i];
            geometry.indices_strides[i] = indices_desc->strides()[i];
        }
    }
};

struct Descriptor::Opaque : public ScatterMooreOpaque {
    using ScatterMooreOpaque::ScatterMooreOpaque;
};

Descriptor::~Descriptor() {
    if (_opaque) {
        delete _opaque;
    }
}

// ==================================================================
// Kernel Launch Logic
// ==================================================================
template <typename T, typename IdxT>
void launch_kernel(
    void *output,
    const void *updates,
    const void *indices,
    const ScatterMooreOpaque *opaque,
    const ScatterInfo &info,
    void *stream) {

    auto out_ptr = reinterpret_cast<T *>(output);
    auto upd_ptr = reinterpret_cast<const T *>(updates);
    auto idx_ptr = reinterpret_cast<const IdxT *>(indices);
    auto musa_stream = reinterpret_cast<musaStream_t>(stream);

    size_t num_updates = 1;
    for (int i = 0; i < opaque->geometry.ndim; ++i) {
        num_updates *= opaque->geometry.updates_shape[i];
    }

    if (num_updates == 0) {
        return;
    }

    size_t block_size = 256;
    size_t grid_size = (num_updates + block_size - 1) / block_size;
    // MUSA grid dimension limit check (usually same as CUDA)
    grid_size = std::min(grid_size, static_cast<size_t>(2147483647));

    op::scatter::moore::scatter_kernel<T, IdxT>
        <<<grid_size, block_size, 0, musa_stream>>>(
            out_ptr,
            upd_ptr,
            idx_ptr,
            opaque->geometry,
            info.axis(),
            info.reduction(),
            num_updates);
}

// ==================================================================
// Descriptor Create
// ==================================================================
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_, Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t indices_desc,
    infiniopTensorDescriptor_t updates_desc,
    int axis,
    int reduction) {

    auto handle = reinterpret_cast<device::moore::Handle *>(handle_);
    auto info_result = ScatterInfo::create(out_desc, input_desc, indices_desc, updates_desc, axis, reduction);
    if (!info_result) {
        return info_result.status();
    }

    if (out_desc->ndim() > op::scatter::moore::MAX_DIMS) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    auto opaque = new Opaque(updates_desc, indices_desc, out_desc);
    size_t workspace_size = 0;

    *desc_ptr = new Descriptor(opaque, info_result.take(), workspace_size, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

// ==================================================================
// Calculate Dispatch
// ==================================================================
infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    const void *indices,
    const void *updates,
    void *stream) const {

    auto musa_stream = reinterpret_cast<musaStream_t>(stream);

    // 1. Copy Input -> Output (if different)
    if (input != output) {
        musaMemcpyAsync(output, input, _opaque->input_bytes, musaMemcpyDeviceToDevice, musa_stream);
    }

    // 2. Launch Kernel
    auto dtype = _info.dtype();
    auto idx_dtype = _info.idx_dtype();

    switch (dtype) {
    case INFINI_DTYPE_F16:
        if (idx_dtype == INFINI_DTYPE_I32) {
            launch_kernel<half, int32_t>(output, updates, indices, _opaque, _info, stream);
        } else {
            launch_kernel<half, int64_t>(output, updates, indices, _opaque, _info, stream);
        }
        break;

    case INFINI_DTYPE_BF16:
        if (idx_dtype == INFINI_DTYPE_I32) {
            launch_kernel<__mt_bfloat16, int32_t>(output, updates, indices, _opaque, _info, stream);
        } else {
            launch_kernel<__mt_bfloat16, int64_t>(output, updates, indices, _opaque, _info, stream);
        }
        break;

    case INFINI_DTYPE_F32:
        if (idx_dtype == INFINI_DTYPE_I32) {
            launch_kernel<float, int32_t>(output, updates, indices, _opaque, _info, stream);
        } else {
            launch_kernel<float, int64_t>(output, updates, indices, _opaque, _info, stream);
        }
        break;

    case INFINI_DTYPE_F64:
        if (idx_dtype == INFINI_DTYPE_I32) {
            launch_kernel<double, int32_t>(output, updates, indices, _opaque, _info, stream);
        } else {
            launch_kernel<double, int64_t>(output, updates, indices, _opaque, _info, stream);
        }
        break;

    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::scatter::moore
