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
    infiniopTensorDescriptor_t cos_desc,
    infiniopRoPEAlgo_t algo) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);

    auto info = RoPEInfo::createRoPEInfo(y_desc, x_desc, pos_desc, sin_desc, cos_desc, algo);
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

    // Calculate position ID stride for batch dimension
    size_t pos_stride_batch = info.pos_has_batch_dim ? info.seqlen : 0;

    // Parallelize over batch and head dimensions - remove collapse clause
#pragma omp parallel for
    for (ptrdiff_t b = 0; b < ptrdiff_t(info.batch); b++) {
        for (ptrdiff_t h = 0; h < ptrdiff_t(info.nhead); h++) {
            for (size_t tok = 0; tok < info.seqlen; tok++) {
                // Calculate memory offsets with batch dimension
                size_t x_offset = (info.has_batch_dim ? b * info.x_stride_batch : 0) + tok * info.x_stride_seqlen + h * info.x_stride_nhead;

                size_t y_offset = (info.has_batch_dim ? b * info.y_stride_batch : 0) + tok * info.y_stride_seqlen + h * info.y_stride_nhead;

                // Calculate position ID offset
                size_t pos_offset;
                if (info.pos_has_batch_dim) {
                    // Per-batch position IDs
                    pos_offset = b * pos_stride_batch + tok;
                } else {
                    // Shared position IDs across batch
                    pos_offset = tok;
                }

                size_t pos_id = size_t(pos_ids[pos_offset]);
                size_t table_offset = pos_id * info.table_dim;

                for (size_t i = 0; i < info.table_dim; i++) {
                    // Calculate positions based on algorithm
                    size_t pos0, pos1;
                    if (info.algo == infiniopRoPEAlgo_t::INFINIOP_ROPE_ALGO_GPT_J) {
                        // GPT-J style: interleaved pairs
                        pos0 = 2 * i;
                        pos1 = 2 * i + 1;
                    } else {
                        // Original style: first half and second half
                        pos0 = i;
                        pos1 = i + info.table_dim;
                    }

                    if constexpr (std::is_same<Tdata, fp16_t>::value || std::is_same<Tdata, bf16_t>::value) {
                        // Convert to float for computation
                        float x0 = utils::cast<float>(x[x_offset + pos0]),
                              x1 = utils::cast<float>(x[x_offset + pos1]),
                              sin__ = utils::cast<float>(sin_table[table_offset + i]),
                              cos__ = utils::cast<float>(cos_table[table_offset + i]);

                        y[y_offset + pos0] = utils::cast<Tdata>(x0 * cos__ - x1 * sin__);
                        y[y_offset + pos1] = utils::cast<Tdata>(x0 * sin__ + x1 * cos__);
                    } else {
                        // Use native types
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
    case INFINI_DTYPE_BF16:
        ROPE_TYPE(bf16_t);
    case INFINI_DTYPE_F32:
        ROPE_TYPE(float);
    case INFINI_DTYPE_F64:
        ROPE_TYPE(double);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

#undef ROPE_TYPE
#undef CALCULATE_ROPE

} // namespace op::rope::cpu
