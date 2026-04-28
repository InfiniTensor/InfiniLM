#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../handle.h"

#include "../cuda/kernel.cuh"
#include "ldexp_nvidia.cuh"
#include <vector>

namespace op::ldexp::nvidia {

// ==================================================================
// Kernel Launch Logic
// ==================================================================

// [修复] 增加 TExp 模板参数，用于处理指数 exp 的不同数据类型
template <typename T, typename TExp>
void launch_kernel(
    void *output,
    const void *x,
    const void *exp,
    const LdexpInfo &info,
    void *stream) {

    // [关键修复] 这里的 T 和 TExp 必须是 CUDA 原生类型 (如 half, cuda_bfloat16)
    // 否则 reinterpret_cast 会导致类型不匹配，且无法在 Device 端使用
    auto out_ptr = reinterpret_cast<T *>(output);
    auto x_ptr = reinterpret_cast<const T *>(x);
    auto exp_ptr = reinterpret_cast<const TExp *>(exp);
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    size_t n = info.count();

    // 填充 KernelShapeInfo
    op::ldexp::cuda::KernelShapeInfo k_info;
    k_info.ndim = info.ndim();
    if (k_info.ndim > op::ldexp::cuda::MAX_DIMS) {
        k_info.ndim = op::ldexp::cuda::MAX_DIMS;
    }

    for (int i = 0; i < k_info.ndim; ++i) {
        k_info.shape[i] = info.shape()[i];
        k_info.stride_x[i] = info.x_strides()[i];
        k_info.stride_exp[i] = info.exp_strides()[i];
    }

    constexpr int block_size = 256;
    size_t grid_size = (n + block_size - 1) / block_size;

    // 启动支持广播的双模板 Kernel
    op::ldexp::cuda::ldexp_broadcast_kernel<T, TExp>
        <<<grid_size, block_size, 0, cuda_stream>>>(
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

    // 显式展开的双层 Switch 分发，每个 case 分 3 行
    // [重要修复] 将所有 fp16_t 替换为 half，bf16_t 替换为 cuda_bfloat16
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
            launch_kernel<float, half>(output, x, exp, _info, stream); // fp16_t -> half
            break;
        case INFINI_DTYPE_BF16:
            launch_kernel<float, cuda_bfloat16>(output, x, exp, _info, stream); // bf16_t -> cuda_bfloat16
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
            launch_kernel<double, half>(output, x, exp, _info, stream); // fp16_t -> half
            break;
        case INFINI_DTYPE_BF16:
            launch_kernel<double, cuda_bfloat16>(output, x, exp, _info, stream); // bf16_t -> cuda_bfloat16
            break;
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        break;

    case INFINI_DTYPE_F16:
        switch (exp_dtype) {
        case INFINI_DTYPE_I32:
            launch_kernel<half, int32_t>(output, x, exp, _info, stream); // fp16_t -> half
            break;
        case INFINI_DTYPE_I64:
            launch_kernel<half, int64_t>(output, x, exp, _info, stream); // fp16_t -> half
            break;
        case INFINI_DTYPE_F32:
            launch_kernel<half, float>(output, x, exp, _info, stream); // fp16_t -> half
            break;
        case INFINI_DTYPE_F16:
            launch_kernel<half, half>(output, x, exp, _info, stream); // fp16_t -> half
            break;
        case INFINI_DTYPE_BF16:
            launch_kernel<half, cuda_bfloat16>(output, x, exp, _info, stream); // fp16_t -> half
            break;
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        break;

    case INFINI_DTYPE_BF16:
        switch (exp_dtype) {
        case INFINI_DTYPE_I32:
            launch_kernel<cuda_bfloat16, int32_t>(output, x, exp, _info, stream); // bf16_t -> cuda_bfloat16
            break;
        case INFINI_DTYPE_I64:
            launch_kernel<cuda_bfloat16, int64_t>(output, x, exp, _info, stream); // bf16_t -> cuda_bfloat16
            break;
        case INFINI_DTYPE_F32:
            launch_kernel<cuda_bfloat16, float>(output, x, exp, _info, stream); // bf16_t -> cuda_bfloat16
            break;
        case INFINI_DTYPE_F16:
            launch_kernel<cuda_bfloat16, half>(output, x, exp, _info, stream); // bf16_t -> cuda_bfloat16
            break;
        case INFINI_DTYPE_BF16:
            launch_kernel<cuda_bfloat16, cuda_bfloat16>(output, x, exp, _info, stream); // bf16_t -> cuda_bfloat16
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

} // namespace op::ldexp::nvidia
