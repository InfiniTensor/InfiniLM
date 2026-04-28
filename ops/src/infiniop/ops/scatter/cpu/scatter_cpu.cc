#include "scatter_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include <algorithm>
#include <cmath>
#include <cstring> // for memcpy
#include <numeric>
#include <vector>

#include "../../../../utils.h"
#include "../../../../utils/custom_types.h"

namespace op::scatter::cpu {
struct ScatterCpuOpaque {
    std::vector<int64_t> updates_shape;
    std::vector<int64_t> updates_strides;
    std::vector<int64_t> output_strides;
    std::vector<int64_t> indices_strides;
    size_t input_total_bytes;

    ScatterCpuOpaque(const infiniopTensorDescriptor_t upd,
                     const infiniopTensorDescriptor_t indices,
                     const infiniopTensorDescriptor_t out) {
        // 1. 几何信息
        const auto &u_shape = upd->shape();
        updates_shape.assign(u_shape.begin(), u_shape.end());

        const auto &u_strides = upd->strides();
        updates_strides.assign(u_strides.begin(), u_strides.end());

        const auto &i_strides = indices->strides();
        indices_strides.assign(i_strides.begin(), i_strides.end()); // <--- 记录 indices strides

        const auto &o_strides = out->strides();
        output_strides.assign(o_strides.begin(), o_strides.end());

        size_t total_elements = 1;
        for (auto s : out->shape()) {
            total_elements *= s;
        }

        size_t dtype_size = 0;
        if (out->dtype() == INFINI_DTYPE_F32) {
            dtype_size = 4;
        } else if (out->dtype() == INFINI_DTYPE_F64) {
            dtype_size = 8;
        } else {
            dtype_size = 2; // f16/bf16
        }

        input_total_bytes = total_elements * dtype_size;
    }
};

struct Descriptor::Opaque : public ScatterCpuOpaque {
    using ScatterCpuOpaque::ScatterCpuOpaque;
};

Descriptor::~Descriptor() {
    if (_opaque) {
        delete _opaque;
        _opaque = nullptr;
    }
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t indices_desc,
    infiniopTensorDescriptor_t updates_desc,
    int axis,
    int reduction) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto result = ScatterInfo::create(out_desc, input_desc, indices_desc, updates_desc, axis, reduction);
    CHECK_RESULT(result);

    // 传入 indices_desc
    auto opaque = new Opaque(updates_desc, indices_desc, out_desc);

    *desc_ptr = new Descriptor(opaque, result.take(), 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

inline void offset_to_coords(int64_t offset, size_t ndim, const int64_t *shape, int64_t *coords) {
    for (size_t i = ndim - 1; i >= 0; --i) {
        coords[i] = offset % shape[i];
        offset /= shape[i];
    }
}

inline int64_t coords_to_offset(size_t ndim, const int64_t *coords, const int64_t *strides) {
    int64_t offset = 0;
    for (size_t i = 0; i < ndim; ++i) {
        offset += coords[i] * strides[i];
    }
    return offset;
}

template <typename T, typename IdxT>
void calculate_cpu_kernel(
    const ScatterInfo &info,
    const ScatterCpuOpaque *opaque,
    void *output,
    const void *indices,
    const void *updates) {

    int axis = info.axis();
    int reduction = info.reduction();
    size_t ndim = info.ndim();

    T *out_ptr = reinterpret_cast<T *>(output);
    const IdxT *idx_ptr = reinterpret_cast<const IdxT *>(indices);
    const T *upd_ptr = reinterpret_cast<const T *>(updates);

    const int64_t *upd_shape_ptr = opaque->updates_shape.data();
    const int64_t *upd_strides_ptr = opaque->updates_strides.data();
    const int64_t *idx_strides_ptr = opaque->indices_strides.data(); // <--- 使用 indices strides
    const int64_t *out_strides_ptr = opaque->output_strides.data();

    size_t total_elements = 1;
    for (auto s : opaque->updates_shape) {
        total_elements *= s;
    }

    // Serial loop
    for (size_t i = 0; i < total_elements; ++i) {
        std::vector<int64_t> coords(ndim);
        offset_to_coords(static_cast<int64_t>(i), ndim, upd_shape_ptr, coords.data());

        int64_t upd_offset = coords_to_offset(ndim, coords.data(), upd_strides_ptr);
        int64_t idx_offset = coords_to_offset(ndim, coords.data(), idx_strides_ptr);

        T upd_val = upd_ptr[upd_offset];
        IdxT idx_val = idx_ptr[idx_offset];

        coords[axis] = static_cast<int64_t>(idx_val);

        int64_t out_offset = coords_to_offset(ndim, coords.data(), out_strides_ptr);

        if (reduction == 0) {
            out_ptr[out_offset] = upd_val;
        } else if (reduction == 1) {
            float val_out = utils::cast<float>(out_ptr[out_offset]);
            float val_upd = utils::cast<float>(upd_val);
            out_ptr[out_offset] = utils::cast<T>(val_out + val_upd);
        } else if (reduction == 2) {
            float val_out = utils::cast<float>(out_ptr[out_offset]);
            float val_upd = utils::cast<float>(upd_val);
            out_ptr[out_offset] = utils::cast<T>(val_out * val_upd);
        }
    }
}

template <typename T>
void calculate_cpu_impl(
    const ScatterInfo &info,
    const ScatterCpuOpaque *opaque,
    void *output,
    const void *input, // 需要 input 指针
    const void *indices,
    const void *updates) {

    if (input != output) {
        std::memcpy(output, input, opaque->input_total_bytes);
    }
    if (info.idx_dtype() == INFINI_DTYPE_I32) {
        calculate_cpu_kernel<T, int32_t>(info, opaque, output, indices, updates);
    } else {
        calculate_cpu_kernel<T, int64_t>(info, opaque, output, indices, updates);
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    const void *indices,
    const void *updates,
    void *stream) const {

    auto dtype = _info.dtype();

    switch (dtype) {
    case INFINI_DTYPE_F32:
        cpu::calculate_cpu_impl<float>(_info, _opaque, output, input, indices, updates);
        break;
    case INFINI_DTYPE_F64:
        cpu::calculate_cpu_impl<double>(_info, _opaque, output, input, indices, updates);
        break;
    case INFINI_DTYPE_F16:
        cpu::calculate_cpu_impl<fp16_t>(_info, _opaque, output, input, indices, updates);
        break;
    case INFINI_DTYPE_BF16:
        cpu::calculate_cpu_impl<bf16_t>(_info, _opaque, output, input, indices, updates);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::scatter::cpu
