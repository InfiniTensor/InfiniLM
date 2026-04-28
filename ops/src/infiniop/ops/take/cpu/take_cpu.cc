#include "take_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include <algorithm>
#include <cstring>
#include <omp.h>

namespace op::take::cpu {

Descriptor::~Descriptor() = default;

// ==================================================================
// 创建描述符
// ==================================================================
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t in_desc,
    infiniopTensorDescriptor_t indices_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);

    auto result = TakeInfo::create(out_desc, in_desc, indices_desc);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        nullptr,       // Opaque*
        result.take(), // Info
        0,             // Workspace Size
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

// ==================================================================
// 核心计算逻辑 (模板实现)
// ==================================================================
template <typename TData, typename TIdx>
void calculate_cpu_impl(
    const TakeInfo &info,
    void *output,
    const void *input,
    const void *indices) {

    size_t num_out = info.num_out();
    size_t num_in = info.num_in();

    auto out_ptr = reinterpret_cast<TData *>(output);
    auto in_ptr = reinterpret_cast<const TData *>(input);
    auto idx_ptr = reinterpret_cast<const TIdx *>(indices);

    // OpenMP 并行化处理
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < (ptrdiff_t)num_out; ++i) {
        TIdx idx = idx_ptr[i];

        // 边界检查
        if (idx >= 0 && static_cast<size_t>(idx) < num_in) {
            out_ptr[i] = in_ptr[idx];
        } else {
            // 越界处理：填充 0
            if constexpr (std::is_arithmetic_v<TData>) {
                // 标准类型 (float, int 等) 直接转换
                out_ptr[i] = static_cast<TData>(0);
            } else {
                out_ptr[i] = utils::cast<TData>(0.0f);
            }
        }
    }
}

// ==================================================================
// 执行函数 (分发逻辑)
// ==================================================================
infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    const void *indices,
    void *stream) const {

    auto dtype = _info.dtype();
    auto idx_dtype = _info.idx_dtype();

// 辅助宏：根据 idx_dtype 分发
#define DISPATCH_IDX(TDATA)                                                     \
    switch (idx_dtype) {                                                        \
    case INFINI_DTYPE_I32:                                                      \
        cpu::calculate_cpu_impl<TDATA, int32_t>(_info, output, input, indices); \
        return INFINI_STATUS_SUCCESS;                                           \
    case INFINI_DTYPE_I64:                                                      \
        cpu::calculate_cpu_impl<TDATA, int64_t>(_info, output, input, indices); \
        return INFINI_STATUS_SUCCESS;                                           \
    default:                                                                    \
        return INFINI_STATUS_BAD_TENSOR_DTYPE;                                  \
    }

    // 主 Switch：根据 dtype 分发
    switch (dtype) {
    // 浮点类型
    case INFINI_DTYPE_F16:
        DISPATCH_IDX(fp16_t);
    case INFINI_DTYPE_BF16:
        DISPATCH_IDX(bf16_t);
    case INFINI_DTYPE_F32:
        DISPATCH_IDX(float);
    case INFINI_DTYPE_F64:
        DISPATCH_IDX(double);

    // 整数类型
    case INFINI_DTYPE_I8:
        DISPATCH_IDX(int8_t);
    case INFINI_DTYPE_U8:
        DISPATCH_IDX(uint8_t);
    case INFINI_DTYPE_I16:
        DISPATCH_IDX(int16_t);
    case INFINI_DTYPE_U16:
        DISPATCH_IDX(uint16_t);
    case INFINI_DTYPE_I32:
        DISPATCH_IDX(int32_t);
    case INFINI_DTYPE_U32:
        DISPATCH_IDX(uint32_t);
    case INFINI_DTYPE_I64:
        DISPATCH_IDX(int64_t);
    case INFINI_DTYPE_U64:
        DISPATCH_IDX(uint64_t);

    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

#undef DISPATCH_IDX
}

} // namespace op::take::cpu
