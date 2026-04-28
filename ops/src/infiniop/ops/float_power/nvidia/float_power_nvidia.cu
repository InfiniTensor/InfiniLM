#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../handle.h"

#include "../cuda/kernel.cuh"
#include "float_power_nvidia.cuh"
#include <algorithm>
#include <cstdint>

namespace op::float_power::nvidia {

// ==================================================================
// 辅助函数: 检查内存地址对齐情况
// ==================================================================
template <typename T>
bool is_aligned(const void *ptr, size_t alignment) {
    return reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
}
template <typename T_OUT, typename T_IN>
void launch_kernel(
    void *output,
    const void *input,
    const void *exponent,
    const FloatPowerInfo &info,
    void *stream) {

    size_t numel = info.num_elements();
    bool is_scalar = info.is_scalar_exponent();
    float scalar_exp = info.scalar_exponent();

    auto out_ptr = reinterpret_cast<T_OUT *>(output);
    auto in_ptr = reinterpret_cast<const T_IN *>(input);
    // 假设指数 Tensor 的数据类型与输入 Tensor 一致
    auto exp_ptr = reinterpret_cast<const T_IN *>(exponent);

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    op::float_power::cuda::FloatPowerFunctor functor;

    // ------------------------------------------------------------------
    // 1. 向量化分发路径 (Vectorized Path)
    // ------------------------------------------------------------------
    constexpr int AlignBytes = 16; // 16字节对齐是 CUDA 访存优化的标准
    constexpr int PackSizeIn = AlignBytes / sizeof(T_IN);
    bool types_same_size = (sizeof(T_IN) == sizeof(T_OUT));

    bool can_vectorize_base = types_same_size && (PackSizeIn > 1) && (numel % PackSizeIn == 0) && is_aligned<T_IN>(input, AlignBytes) && is_aligned<T_OUT>(output, AlignBytes);

    if (can_vectorize_base) {
        size_t num_packs = numel / PackSizeIn;
        size_t block_size = 256;
        size_t grid_size = (num_packs + block_size - 1) / block_size;

        if (is_scalar) {
            // 路径 A1: 标量指数向量化（极快）
            op::float_power::cuda::float_power_kernel_vectorized_scalar<T_OUT, T_IN, PackSizeIn>
                <<<grid_size, block_size, 0, cuda_stream>>>(
                    out_ptr, in_ptr, scalar_exp, num_packs, functor);
            return;
        } else if (is_aligned<T_IN>(exponent, AlignBytes)) {
            // 路径 A2: 张量指数向量化（解决 0.2x 倍速问题的核心）
            op::float_power::cuda::float_power_kernel_vectorized_tensor<T_OUT, T_IN, PackSizeIn>
                <<<grid_size, block_size, 0, cuda_stream>>>(
                    out_ptr, in_ptr, exp_ptr, num_packs, functor);
            return;
        }
    }

    // ------------------------------------------------------------------
    // 2. 通用回退路径 (Fallback Path)
    // 处理不对齐、非对称类型转换或小规模数据的场景
    // ------------------------------------------------------------------
    size_t block_size = 256;
    size_t grid_size = (numel + block_size - 1) / block_size;

    op::float_power::cuda::float_power_kernel<T_OUT, T_IN, T_IN>
        <<<grid_size, block_size, 0, cuda_stream>>>(
            out_ptr, in_ptr, exp_ptr, scalar_exp, is_scalar, numel, functor);
}

// ==================================================================
// Descriptor 接口实现
// ==================================================================
struct Descriptor::Opaque {};

Descriptor::~Descriptor() {
    if (_opaque) {
        delete _opaque;
    }
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle, Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y,
    infiniopTensorDescriptor_t x,
    infiniopTensorDescriptor_t exponent,
    float scalar_exponent) {

    auto info_result = FloatPowerInfo::create(y, x, exponent, scalar_exponent);
    if (!info_result) {
        return info_result.status();
    }

    size_t workspace_size = 0;
    *desc_ptr = new Descriptor(new Opaque(), info_result.take(), workspace_size, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size, void *output,
    const void *input, const void *exponent,
    void *stream) const {

    auto in_dtype = _info.input_dtype();
    auto out_dtype = _info.output_dtype();

    // ==================================================================
    // 完全显式双重分发 (Fully Explicit Double Dispatch)
    // ==================================================================

    switch (in_dtype) {

    case INFINI_DTYPE_F32:
        switch (out_dtype) {
        case INFINI_DTYPE_F32:
            launch_kernel<float, float>(output, input, exponent, _info, stream);
            break;
        case INFINI_DTYPE_F64:
            launch_kernel<double, float>(output, input, exponent, _info, stream);
            break;
        case INFINI_DTYPE_F16:
            launch_kernel<half, float>(output, input, exponent, _info, stream);
            break;
        case INFINI_DTYPE_BF16:
            launch_kernel<cuda_bfloat16, float>(output, input, exponent, _info, stream);
            break;
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        break;

    case INFINI_DTYPE_F64:
        switch (out_dtype) {
        case INFINI_DTYPE_F32:
            launch_kernel<float, double>(output, input, exponent, _info, stream);
            break;
        case INFINI_DTYPE_F64:
            launch_kernel<double, double>(output, input, exponent, _info, stream);
            break;
        case INFINI_DTYPE_F16:
            launch_kernel<half, double>(output, input, exponent, _info, stream);
            break;
        case INFINI_DTYPE_BF16:
            launch_kernel<cuda_bfloat16, double>(output, input, exponent, _info, stream);
            break;
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        break;

    case INFINI_DTYPE_F16:
        switch (out_dtype) {
        case INFINI_DTYPE_F32:
            launch_kernel<float, half>(output, input, exponent, _info, stream);
            break;
        case INFINI_DTYPE_F64:
            launch_kernel<double, half>(output, input, exponent, _info, stream);
            break;
        case INFINI_DTYPE_F16:
            launch_kernel<half, half>(output, input, exponent, _info, stream);
            break;
        case INFINI_DTYPE_BF16:
            launch_kernel<cuda_bfloat16, half>(output, input, exponent, _info, stream);
            break;
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        break;

    case INFINI_DTYPE_BF16:
        switch (out_dtype) {
        case INFINI_DTYPE_F32:
            launch_kernel<float, cuda_bfloat16>(output, input, exponent, _info, stream);
            break;
        case INFINI_DTYPE_F64:
            launch_kernel<double, cuda_bfloat16>(output, input, exponent, _info, stream);
            break;
        case INFINI_DTYPE_F16:
            launch_kernel<half, cuda_bfloat16>(output, input, exponent, _info, stream);
            break;
        case INFINI_DTYPE_BF16:
            launch_kernel<cuda_bfloat16, cuda_bfloat16>(output, input, exponent, _info, stream);
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
} // namespace op::float_power::nvidia
