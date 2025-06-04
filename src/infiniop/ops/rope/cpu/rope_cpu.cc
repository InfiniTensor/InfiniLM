#include "rope_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

namespace op::rope::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t pos_desc,
    infiniopTensorDescriptor_t sin_desc,
    infiniopTensorDescriptor_t cos_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);

    auto info = RoPEInfo::createRoPEInfo(y_desc, x_desc, pos_desc, sin_desc, cos_desc);
    CHECK_RESULT(info);

    // Create descriptor
    *desc_ptr = new Descriptor(
        info.take(),
        0,
        nullptr,
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata, typename Tindex>
infiniStatus_t calculateRoPE(const RoPEInfo &info,
                             Tdata *y,
                             const Tdata *x,
                             const Tindex *pos_ids,
                             const Tdata *sin_table,
                             const Tdata *cos_table) {
#pragma omp parallel for
    for (ptrdiff_t h = 0; h < ptrdiff_t(info.nhead); h++) {
        for (size_t tok = 0; tok < info.seqlen; tok++) {
            size_t x_offset = tok * info.x_stride_seqlen + h * info.x_stride_nhead;
            size_t y_offset = tok * info.y_stride_seqlen + h * info.y_stride_nhead;
            size_t pos_id = size_t(pos_ids[tok]);
            size_t table_offset = pos_id * info.table_dim;

            for (size_t i = 0; i < info.table_dim; i++) {
                size_t pos0 = 2 * i;
                size_t pos1 = 2 * i + 1;

                if constexpr (std::is_same<Tdata, fp16_t>::value) {
                    float x0 = utils::cast<float>(x[x_offset + pos0]),
                          x1 = utils::cast<float>(x[x_offset + pos1]),
                          sin__ = utils::cast<float>(sin_table[table_offset + i]),
                          cos__ = utils::cast<float>(cos_table[table_offset + i]);

                    y[y_offset + pos0] = utils::cast<fp16_t>(x0 * cos__ - x1 * sin__);
                    y[y_offset + pos1] = utils::cast<fp16_t>(x0 * sin__ + x1 * cos__);
                } else {
                    Tdata x0 = x[x_offset + pos0],
                          x1 = x[x_offset + pos1],
                          sin__ = sin_table[table_offset + i],
                          cos__ = cos_table[table_offset + i];

                    y[y_offset + pos0] = x0 * cos__ - x1 * sin__;
                    y[y_offset + pos1] = x0 * sin__ + x1 * cos__;
                }
            }
        }
    }

    return INFINI_STATUS_SUCCESS;
}

#define CALCULATE_ROPE(TDATA, TINDEX) \
    calculateRoPE(_info, (TDATA *)y, (const TDATA *)x, (const TINDEX *)pos_ids, (const TDATA *)sin_table, (const TDATA *)cos_table)

#define ROPE_TYPE(TDATA)                        \
    switch (_info.pos_type) {                   \
    case INFINI_DTYPE_U8:                       \
        return CALCULATE_ROPE(TDATA, uint8_t);  \
    case INFINI_DTYPE_U16:                      \
        return CALCULATE_ROPE(TDATA, uint16_t); \
    case INFINI_DTYPE_U32:                      \
        return CALCULATE_ROPE(TDATA, uint32_t); \
    case INFINI_DTYPE_U64:                      \
        return CALCULATE_ROPE(TDATA, uint64_t); \
    case INFINI_DTYPE_I8:                       \
        return CALCULATE_ROPE(TDATA, int8_t);   \
    case INFINI_DTYPE_I16:                      \
        return CALCULATE_ROPE(TDATA, int16_t);  \
    case INFINI_DTYPE_I32:                      \
        return CALCULATE_ROPE(TDATA, int32_t);  \
    case INFINI_DTYPE_I64:                      \
        return CALCULATE_ROPE(TDATA, int64_t);  \
    default:                                    \
        return INFINI_STATUS_BAD_TENSOR_DTYPE;  \
    }

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    const void *pos_ids,
    const void *sin_table,
    const void *cos_table,
    void *stream) const {

    switch (_info.data_type) {
    case INFINI_DTYPE_F16:
        ROPE_TYPE(fp16_t);
    case INFINI_DTYPE_F32:
        ROPE_TYPE(float);
    case INFINI_DTYPE_F64:
        ROPE_TYPE(double);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

#undef ROPE_TYPE
#undef CALCULATE_ROPE

} // namespace op::rope::cpu
