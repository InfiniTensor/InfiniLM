#include "rope_aclnn.h"
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

extern "C" infiniStatus_t rope_kernel_launch(
    void *y,
    void *x,
    void *pos,
    void *sin,
    void *cos,
    int32_t seq_len,
    int32_t nhead,
    int32_t dhead,
    int32_t data_type,
    int32_t y_stride_seqlen,
    int32_t y_stride_nhead,
    int32_t x_stride_seqlen,
    int32_t x_stride_nhead,
    void *stream);

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    const void *pos_ids,
    const void *sin_table,
    const void *cos_table,
    void *stream) const {
    // TODO: 是否强加这个判断
    std::cout << "pos_type: " << _info.pos_type << std::endl;
    CHECK_DTYPE(_info.pos_type, INFINI_DTYPE_U32);
    CHECK_DTYPE(_info.data_type, INFINI_DTYPE_F32, INFINI_DTYPE_F16);
    int32_t seq_len = _info.seqlen;
    int32_t nhead = _info.nhead;
    int32_t dhead = _info.dhead;
    int32_t data_type = _info.data_type;
    int32_t y_stride_seqlen = _info.y_stride_seqlen;
    int32_t y_stride_nhead = _info.y_stride_nhead;
    int32_t x_stride_seqlen = _info.x_stride_seqlen;
    int32_t x_stride_nhead = _info.x_stride_nhead;
    std::cout << "shape is " << seq_len << ", " << nhead << ", " << dhead << std::endl;
    std::cout << "y_stride is " << y_stride_seqlen << ", " << y_stride_nhead << ", 1" << std::endl;
    std::cout << "x_stride is " << x_stride_seqlen << ", " << x_stride_nhead << ", 1" << std::endl;
    return rope_kernel_launch(y, (void *)x, (void *)pos_ids, (void *)sin_table, (void *)cos_table, seq_len, nhead, dhead, data_type, y_stride_seqlen, y_stride_nhead, x_stride_seqlen, x_stride_nhead, stream);
}
} // namespace op::rope::ascend
