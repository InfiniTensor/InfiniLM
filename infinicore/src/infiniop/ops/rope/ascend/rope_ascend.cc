#include "rope_ascend.h"
#include "../../../devices/ascend/common_ascend.h"

namespace op::rope::ascend {

Descriptor::~Descriptor()
    = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t pos_desc,
    infiniopTensorDescriptor_t sin_desc,
    infiniopTensorDescriptor_t cos_desc) {
    auto handle_ascned = reinterpret_cast<device::ascend::Handle *>(handle);
    auto result = RoPEInfo::createRoPEInfo(y_desc, x_desc, pos_desc, sin_desc, cos_desc);
    CHECK_RESULT(result);

    size_t workspace_size = 0;
    *desc_ptr = new Descriptor(std::move(result.take()), workspace_size, nullptr, handle_ascned->device, handle_ascned->device_id);
    return INFINI_STATUS_SUCCESS;
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
    CHECK_DTYPE(_info.data_type, INFINI_DTYPE_F32, INFINI_DTYPE_F16);

    auto data_type = _info.data_type;
    auto pos_type = _info.pos_type;
    auto seq_len = _info.seqlen;
    auto nhead = _info.nhead;
    auto dhead = _info.dhead;

    auto y_stride_seqlen = _info.y_stride_seqlen;
    auto y_stride_nhead = _info.y_stride_nhead;
    auto x_stride_seqlen = _info.x_stride_seqlen;
    auto x_stride_nhead = _info.x_stride_nhead;

    return rope_kernel_launch(y, (void *)x, (void *)pos_ids, (void *)sin_table, (void *)cos_table, seq_len, nhead, dhead, data_type, pos_type, y_stride_seqlen, y_stride_nhead, x_stride_seqlen, x_stride_nhead, stream);
}
} // namespace op::rope::ascend
