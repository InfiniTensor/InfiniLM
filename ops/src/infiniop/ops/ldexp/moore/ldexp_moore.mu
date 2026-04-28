#include "../../../devices/moore/moore_handle.h"
#include "../../../handle.h"
#include "ldexp_moore.h"
#include "ldexp_moore_kernel.h"
#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <vector>

namespace op::ldexp::moore {

// ==================================================================
// Kernel Launch Logic
// ==================================================================

template <typename T, typename TExp>
void launch_kernel(
    void *output,
    const void *x,
    const void *exp,
    const LdexpInfo &info,
    void *stream) {

    auto out_ptr = reinterpret_cast<T *>(output);
    auto x_ptr = reinterpret_cast<const T *>(x);
    auto exp_ptr = reinterpret_cast<const TExp *>(exp);
    auto musa_stream = reinterpret_cast<musaStream_t>(stream);

    size_t n = info.count();

    op::ldexp::moore::KernelShapeInfo k_info;
    k_info.ndim = info.ndim();
    if (k_info.ndim > op::ldexp::moore::MAX_DIMS) {
        k_info.ndim = op::ldexp::moore::MAX_DIMS;
    }

    for (int i = 0; i < k_info.ndim; ++i) {
        k_info.shape[i] = info.shape()[i];
        k_info.stride_x[i] = info.x_strides()[i];
        k_info.stride_exp[i] = info.exp_strides()[i];
    }

    constexpr int block_size = 256;
    size_t grid_size = (n + block_size - 1) / block_size;
    if (grid_size > 65535) {
        grid_size = 65535;
    }

    op::ldexp::moore::ldexp_broadcast_kernel<T, TExp>
        <<<grid_size, block_size, 0, musa_stream>>>(
            out_ptr, x_ptr, exp_ptr, n, k_info);
}

// ==================================================================
// Descriptor Implementation
// ==================================================================
struct Descriptor::Opaque {};

Descriptor::~Descriptor() {
    if (_opaque) {
        delete _opaque;
    }
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t exp_desc) {

    auto info_result = LdexpInfo::create(y_desc, x_desc, exp_desc);
    if (!info_result) {
        return info_result.status();
    }

    size_t workspace_size = 0;

    *desc_ptr = new Descriptor(
        new Opaque(),
        info_result.take(),
        workspace_size,
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const {

    if (inputs.size() != 2) {
        return INFINI_STATUS_BAD_PARAM;
    }
    return calculate(workspace, workspace_size, output, inputs[0], inputs[1], stream);
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *x,
    const void *exp,
    void *stream) const {

    auto dtype = _info.dtype();
    auto exp_dtype = _info.exp_dtype();

    switch (dtype) {
    case INFINI_DTYPE_F32:
        switch (exp_dtype) {
        case INFINI_DTYPE_I32:
            launch_kernel<float, int32_t>(output, x, exp, _info, stream);
            break;
        case INFINI_DTYPE_I64:
            launch_kernel<float, int64_t>(output, x, exp, _info, stream);
            break;
        case INFINI_DTYPE_F32:
            launch_kernel<float, float>(output, x, exp, _info, stream);
            break;
        case INFINI_DTYPE_F16:
            launch_kernel<float, half>(output, x, exp, _info, stream);
            break;
        case INFINI_DTYPE_BF16:
            launch_kernel<float, __mt_bfloat16>(output, x, exp, _info, stream);
            break;
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        break;

    case INFINI_DTYPE_F64:
        switch (exp_dtype) {
        case INFINI_DTYPE_I32:
            launch_kernel<double, int32_t>(output, x, exp, _info, stream);
            break;
        case INFINI_DTYPE_I64:
            launch_kernel<double, int64_t>(output, x, exp, _info, stream);
            break;
        case INFINI_DTYPE_F32:
            launch_kernel<double, float>(output, x, exp, _info, stream);
            break;
        case INFINI_DTYPE_F16:
            launch_kernel<double, half>(output, x, exp, _info, stream);
            break;
        case INFINI_DTYPE_BF16:
            launch_kernel<double, __mt_bfloat16>(output, x, exp, _info, stream);
            break;
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        break;

    case INFINI_DTYPE_F16:
        switch (exp_dtype) {
        case INFINI_DTYPE_I32:
            launch_kernel<half, int32_t>(output, x, exp, _info, stream);
            break;
        case INFINI_DTYPE_I64:
            launch_kernel<half, int64_t>(output, x, exp, _info, stream);
            break;
        case INFINI_DTYPE_F32:
            launch_kernel<half, float>(output, x, exp, _info, stream);
            break;
        case INFINI_DTYPE_F16:
            launch_kernel<half, half>(output, x, exp, _info, stream);
            break;
        case INFINI_DTYPE_BF16:
            launch_kernel<half, __mt_bfloat16>(output, x, exp, _info, stream);
            break;
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        break;

    case INFINI_DTYPE_BF16:
        switch (exp_dtype) {
        case INFINI_DTYPE_I32:
            launch_kernel<__mt_bfloat16, int32_t>(output, x, exp, _info, stream);
            break;
        case INFINI_DTYPE_I64:
            launch_kernel<__mt_bfloat16, int64_t>(output, x, exp, _info, stream);
            break;
        case INFINI_DTYPE_F32:
            launch_kernel<__mt_bfloat16, float>(output, x, exp, _info, stream);
            break;
        case INFINI_DTYPE_F16:
            launch_kernel<__mt_bfloat16, half>(output, x, exp, _info, stream);
            break;
        case INFINI_DTYPE_BF16:
            launch_kernel<__mt_bfloat16, __mt_bfloat16>(output, x, exp, _info, stream);
            break;
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        break;

    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::ldexp::moore
