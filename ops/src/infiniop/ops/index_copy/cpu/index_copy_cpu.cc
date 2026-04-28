#include "index_copy_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include <algorithm>
#include <cstring>
#include <vector>

namespace op::index_copy::cpu {

Descriptor::~Descriptor() = default;

// ==================================================================
// 创建描述符
// ==================================================================
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t in_desc,
    int64_t dim,
    infiniopTensorDescriptor_t index_desc,
    infiniopTensorDescriptor_t source_desc) { // 注意：移除了 float alpha

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);

    // 创建 Info 对象
    auto result = IndexCopyInfo::create(out_desc, in_desc, dim, index_desc, source_desc);
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
// 核心计算逻辑 (串行实现)
// ==================================================================
template <typename TData, typename TIdx>
void calculate_cpu_impl(
    const IndexCopyInfo &info,
    void *output,
    const void *input,
    const void *index,
    const void *source) {

    // IndexCopy 不需要 Alpha 也不需要提升精度进行计算，直接拷贝即可

    // 1. 获取几何信息
    size_t outer_size = info.outer_size();
    size_t inner_size = info.inner_size();
    size_t dim_size = info.dim_size();
    size_t index_len = info.index_len();

    auto out_ptr = reinterpret_cast<TData *>(output);
    auto src_ptr = reinterpret_cast<const TData *>(source);
    auto idx_ptr = reinterpret_cast<const TIdx *>(index);

    // -----------------------------------------------------------
    // 串行循环逻辑
    // -----------------------------------------------------------
    for (size_t o = 0; o < outer_size; ++o) {
        for (size_t i = 0; i < index_len; ++i) {

            TIdx idx = idx_ptr[i];

            // 处理负索引
            if (idx < 0) {
                idx += static_cast<TIdx>(dim_size);
            }

            // 边界检查
            if (idx < 0 || static_cast<size_t>(idx) >= dim_size) {
                continue;
            }

            // 计算偏移
            size_t src_offset = o * index_len * inner_size + i * inner_size;
            size_t out_offset = o * dim_size * inner_size + static_cast<size_t>(idx) * inner_size;

            // Inner 维度循环
            for (size_t in = 0; in < inner_size; ++in) {
                // 【核心逻辑】
                // IndexCopy: output[idx] = source[i]
                // 直接赋值，无需 utils::cast 提升精度
                out_ptr[out_offset + in] = src_ptr[src_offset + in];
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
    const void *index,
    const void *source,
    void *stream) const {

    auto dtype = _info.dtype();
    auto idx_dtype = _info.idx_dtype();

#define DISPATCH(TDATA, TIDX)                                             \
    calculate_cpu_impl<TDATA, TIDX>(_info, output, input, index, source); \
    return INFINI_STATUS_SUCCESS

    if (idx_dtype == INFINI_DTYPE_I32) {
        switch (dtype) {
        case INFINI_DTYPE_F32:
            DISPATCH(float, int32_t);
        case INFINI_DTYPE_F64:
            DISPATCH(double, int32_t);
        case INFINI_DTYPE_F16:
            DISPATCH(fp16_t, int32_t);
        case INFINI_DTYPE_BF16:
            DISPATCH(bf16_t, int32_t);
        case INFINI_DTYPE_I32:
            DISPATCH(int32_t, int32_t);
        case INFINI_DTYPE_I64:
            DISPATCH(int64_t, int32_t);
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    } else if (idx_dtype == INFINI_DTYPE_I64) {
        switch (dtype) {
        case INFINI_DTYPE_F32:
            DISPATCH(float, int64_t);
        case INFINI_DTYPE_F64:
            DISPATCH(double, int64_t);
        case INFINI_DTYPE_F16:
            DISPATCH(fp16_t, int64_t);
        case INFINI_DTYPE_BF16:
            DISPATCH(bf16_t, int64_t);
        case INFINI_DTYPE_I32:
            DISPATCH(int32_t, int64_t);
        case INFINI_DTYPE_I64:
            DISPATCH(int64_t, int64_t);
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    }

    return INFINI_STATUS_BAD_TENSOR_DTYPE;

#undef DISPATCH
}

} // namespace op::index_copy::cpu
