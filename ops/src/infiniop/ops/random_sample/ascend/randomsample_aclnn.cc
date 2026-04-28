#include "../../../devices/ascend/common_ascend.h"
#include "random_sample_aclnn.h"
#include <aclnnop/aclnn_topk.h>

namespace op::random_sample::ascend {

struct Descriptor::Opaque {
    aclnnTensorDescriptor_t probs;
    aclnnTensorDescriptor_t result;

    ~Opaque() {
        delete probs;
        delete result;
    }
};

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t result_desc,
    infiniopTensorDescriptor_t probs_desc) {
    auto handle = reinterpret_cast<device::ascend::Handle *>(handle_);
    auto result = RandomSampleInfo::create(result_desc, probs_desc);
    CHECK_RESULT(result);
    CHECK_DTYPE(result->dt_i, INFINI_DTYPE_I64);
    auto workspace_size = probs_desc->numel() * infiniSizeOf(probs_desc->dtype()) + probs_desc->numel() * infiniSizeOf(infiniDtype_t::INFINI_DTYPE_I64);
    auto tresult = new aclnnTensorDescriptor(result_desc);
    auto tprobs = new aclnnTensorDescriptor(probs_desc);
    *desc_ptr
        = new Descriptor(
            result.take(),
            workspace_size,
            new Opaque{tprobs, tresult},
            handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

size_t Descriptor::minWorkspaceSize() const {
    return _min_workspace_size;
}

extern "C" infiniStatus_t random_sample_kernel_launch(
    void *probs,
    void *result,
    void *topk_val_addr,
    void *topk_idx_addr,
    float random_val,
    float topp,
    int topk,
    float temperature,
    uint64_t n,
    infiniDtype_t dt_p,
    void *stream);

infiniStatus_t
Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *result,
    const void *probs,
    float random_val,
    float topp,
    int topk,
    float temperature,
    void *stream) const {
    if (workspace_size < _min_workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    auto topk_ = topk <= (int)_info.n ? topk : (int)_info.n;
    bool dosample = topk_ > 1 && temperature != 0.0f && topp != 0.0f && random_val != 0.0f;
    auto topk_shape = std::vector<int64_t>{dosample ? topk_ : 1};
    auto topk_stride = std::vector<int64_t>{1};
    auto topk_idx = new aclnnTensorDescriptor(toAclDataType(_info.dt_i), topk_shape, topk_stride);
    auto topk_val = new aclnnTensorDescriptor(toAclDataType(_info.dt_p), topk_shape, topk_stride);
    auto topk_val_addr = workspace;
    auto topk_idx_addr = (void *)((uint8_t *)workspace + topk_ * infiniSizeOf(_info.dt_p));
    uint64_t topk_workspace_size = 0;
    aclOpExecutor *topk_executor = nullptr;
    CHECK_ACL(aclnnTopkGetWorkspaceSize(_opaque->probs->tensor,
                                        topk_shape[0],
                                        0,
                                        true,
                                        true,
                                        topk_val->tensor,
                                        dosample ? topk_idx->tensor : _opaque->result->tensor,
                                        &topk_workspace_size,
                                        &topk_executor));
    CHECK_ACL(aclSetAclOpExecutorRepeatable(topk_executor));
    void *topk_workspace;
    CHECK_ACL(aclrtMalloc(&topk_workspace, topk_workspace_size, ACL_MEM_MALLOC_HUGE_FIRST));
    AclSetTensorAddr(topk_executor, 0, _opaque->probs->tensor, (void *)probs);
    AclSetTensorAddr(topk_executor, 1, topk_val->tensor, topk_val_addr);
    if (!dosample) {
        AclSetTensorAddr(topk_executor, 2, _opaque->result->tensor, result);
    } else {
        AclSetTensorAddr(topk_executor, 2, topk_idx->tensor, topk_idx_addr);
    }
    CHECK_ACL(aclnnTopk(topk_workspace, topk_workspace_size, topk_executor, stream));
    CHECK_ACL(aclrtFree(topk_workspace));

    if (dosample) {
        auto status = random_sample_kernel_launch((void *)probs, result, topk_val_addr, topk_idx_addr, random_val, topp, topk_, temperature, _info.n, _info.dt_p, stream);
        CHECK_STATUS(status);
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::random_sample::ascend
