#include "swiglu_ascend.h"
#include "../../../devices/ascend/common_ascend.h"

namespace op::swiglu::ascend {
Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(infiniopHandle_t handle, Descriptor **desc_ptr,
                                  infiniopTensorDescriptor_t c_desc,
                                  std::vector<infiniopTensorDescriptor_t> input_descs) {
    auto handle_ascend = reinterpret_cast<device::ascend::Handle *>(handle);

    auto dtype = c_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32);

    const auto &a_desc = input_descs[0];
    const auto &b_desc = input_descs[1];

    auto result = SwigluInfo::create(c_desc, a_desc, b_desc);
    CHECK_RESULT(result);
    SwigluInfo info = result.take();

    // https://www.hiascend.com/document/detail/zh/canncommercial/800/apiref/ascendcopapi/atlasascendc_api_07_0777.html
    size_t workspace_size = 0;

    *desc_ptr = new Descriptor(std::move(info), workspace_size, handle_ascend->device, handle_ascend->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *workspace,
                                     size_t workspace_size,
                                     void *c,
                                     std::vector<const void *> inputs,
                                     void *stream) const {
    auto batch = _info.ndim == 2 ? 1 : _info.shape[0];
    auto seq_len = _info.ndim == 2 ? _info.shape[0] : _info.shape[1];
    auto hidden_size = _info.shape[_info.ndim - 1];
    auto stride_batch_c = _info.ndim == 2 ? 1 : _info.c_strides[0];
    auto stride_batch_a = _info.ndim == 2 ? 1 : _info.a_strides[0];
    auto stride_batch_b = _info.ndim == 2 ? 1 : _info.b_strides[0];
    auto stride_seq_c = _info.ndim == 2 ? _info.c_strides[0] : _info.c_strides[1];
    auto stride_seq_a = _info.ndim == 2 ? _info.a_strides[0] : _info.a_strides[1];
    auto stride_seq_b = _info.ndim == 2 ? _info.b_strides[0] : _info.b_strides[1];
    auto status = swiglu_kernel_launch(c, (void *)inputs[0], (void *)inputs[1], _info.dtype, batch, seq_len, hidden_size, stride_batch_c, stride_batch_a, stride_batch_b, stride_seq_c, stride_seq_a, stride_seq_b, stream);
    return status;
}

} // namespace op::swiglu::ascend
