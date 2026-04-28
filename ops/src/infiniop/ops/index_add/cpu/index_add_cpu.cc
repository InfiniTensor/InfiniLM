#include "index_add_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include <algorithm>
#include <cstring>
#include <type_traits> // 必需：用于 std::conditional
#include <vector>

namespace op::index_add::cpu {

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
    infiniopTensorDescriptor_t source_desc,
    float alpha) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);

    auto result = IndexAddInfo::create(out_desc, in_desc, dim, index_desc, source_desc, alpha);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        nullptr,       // Opaque*
        result.take(), // Info
        0,             // Workspace Size
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

template <typename TData, typename TIdx>
void calculate_cpu_impl(
    const IndexAddInfo &info,
    void *output,
    const void *input,
    const void *index,
    const void *source) {

    using CalcType = typename std::conditional<std::is_same<TData, double>::value, double, float>::type;

    // 转换 Alpha (使用 utils::cast 处理自定义类型)
    CalcType alpha_val = utils::cast<CalcType>(info.alpha());
    size_t outer_size = info.outer_size();
    size_t inner_size = info.inner_size();
    size_t dim_size = info.dim_size();
    size_t index_len = info.index_len();

    auto out_ptr = reinterpret_cast<TData *>(output);
    auto src_ptr = reinterpret_cast<const TData *>(source);
    auto idx_ptr = reinterpret_cast<const TIdx *>(index);

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
                CalcType src_val = utils::cast<CalcType>(src_ptr[src_offset + in]);
                CalcType out_old_val = utils::cast<CalcType>(out_ptr[out_offset + in]);

                // 2. 执行计算: out = out + src * alpha
                CalcType result_val = out_old_val + src_val * alpha_val;

                // 3. 使用 utils::cast 转回 TData 并写入
                out_ptr[out_offset + in] = utils::cast<TData>(result_val);
            }
        }
    }
}

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

} // namespace op::index_add::cpu
