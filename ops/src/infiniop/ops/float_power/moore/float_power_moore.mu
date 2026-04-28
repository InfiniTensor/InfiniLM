#include "../../../devices/moore/moore_handle.h"
#include "float_power_moore.h"
#include "float_power_moore_kernel.h"
#include <algorithm>
#include <cstdint>

namespace op::float_power::moore {

// ==================================================================
// 辅助函数: 检查内存地址对齐情况
// ==================================================================
template <typename T>
bool is_aligned(const void *ptr, size_t alignment) {
    return reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
}

// ==================================================================
// Kernel Launch Logic
// ==================================================================
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
    auto exp_ptr = reinterpret_cast<const T_IN *>(exponent);

    auto musa_stream = reinterpret_cast<musaStream_t>(stream);
    op::float_power::moore::FloatPowerFunctor functor;

    // ------------------------------------------------------------------
    // 1. 向量化分发路径 (Vectorized Path)
    // ------------------------------------------------------------------
    constexpr int AlignBytes = 16;
    constexpr int PackSizeIn = AlignBytes / sizeof(T_IN);

    // 只有当输入和输出类型大小相同时，当前的 1:1 Pack 向量化逻辑才生效
    bool types_same_size = (sizeof(T_IN) == sizeof(T_OUT));

    bool can_vectorize_base = types_same_size && (PackSizeIn > 1) && (numel % PackSizeIn == 0) && is_aligned<T_IN>(input, AlignBytes) && is_aligned<T_OUT>(output, AlignBytes);

    if (can_vectorize_base) {
        size_t num_packs = numel / PackSizeIn;
        size_t block_size = 256;
        size_t grid_size = (num_packs + block_size - 1) / block_size;

        if (is_scalar) {
            // 路径 A1: 标量指数向量化
            op::float_power::moore::float_power_kernel_vectorized_scalar<T_OUT, T_IN, PackSizeIn>
                <<<grid_size, block_size, 0, musa_stream>>>(
                    out_ptr, in_ptr, scalar_exp, num_packs, functor);
            return;
        } else if (is_aligned<T_IN>(exponent, AlignBytes)) {
            // 路径 A2: 张量指数向量化
            op::float_power::moore::float_power_kernel_vectorized_tensor<T_OUT, T_IN, PackSizeIn>
                <<<grid_size, block_size, 0, musa_stream>>>(
                    out_ptr, in_ptr, exp_ptr, num_packs, functor);
            return;
        }
    }

    // ------------------------------------------------------------------
    // 2. 通用回退路径 (Fallback Path)
    // ------------------------------------------------------------------
    size_t block_size = 256;
    size_t grid_size = (numel + block_size - 1) / block_size;

    op::float_power::moore::float_power_kernel<T_OUT, T_IN, T_IN>
        <<<grid_size, block_size, 0, musa_stream>>>(
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
    infiniopHandle_t handle_, Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y,
    infiniopTensorDescriptor_t x,
    infiniopTensorDescriptor_t exponent,
    float scalar_exponent) {

    auto handle = reinterpret_cast<device::moore::Handle *>(handle_);

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
            launch_kernel<__mt_bfloat16, float>(output, input, exponent, _info, stream);
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
            launch_kernel<__mt_bfloat16, double>(output, input, exponent, _info, stream);
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
            launch_kernel<__mt_bfloat16, half>(output, input, exponent, _info, stream);
            break;
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        break;

    case INFINI_DTYPE_BF16:
        switch (out_dtype) {
        case INFINI_DTYPE_F32:
            launch_kernel<float, __mt_bfloat16>(output, input, exponent, _info, stream);
            break;
        case INFINI_DTYPE_F64:
            launch_kernel<double, __mt_bfloat16>(output, input, exponent, _info, stream);
            break;
        case INFINI_DTYPE_F16:
            launch_kernel<half, __mt_bfloat16>(output, input, exponent, _info, stream);
            break;
        case INFINI_DTYPE_BF16:
            launch_kernel<__mt_bfloat16, __mt_bfloat16>(output, input, exponent, _info, stream);
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
} // namespace op::float_power::moore
