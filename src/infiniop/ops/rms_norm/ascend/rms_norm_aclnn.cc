#include "rms_norm_aclnn.h"
#include "../../../devices/ascend/common_ascend.h"
#include <aclnnop/aclnn_rms_norm.h>

namespace op::rms_norm::ascend {

struct Descriptor::Opaque {
    mutable aclOpExecutor *executor;
    aclnnTensorDescriptor_t y;
    aclnnTensorDescriptor_t x;
    aclnnTensorDescriptor_t w;
    aclnnTensorDescriptor_t rstd;
    size_t workspaceSize;

    ~Opaque() {
        delete y;
        delete x;
        delete w;
        delete rstd;
        aclDestroyAclOpExecutor(executor);
    }
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t w_desc,
    float epsilon) {

    auto result = RMSNormInfo::create(y_desc, x_desc, w_desc, epsilon);
    CHECK_RESULT(result);
    auto info = result.take();

    size_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnTensorDescriptor_t y = nullptr;
    aclnnTensorDescriptor_t x = nullptr;
    aclnnTensorDescriptor_t w = nullptr;
    aclnnTensorDescriptor_t rstd = nullptr;

    std::vector<int64_t> slice_shape = {static_cast<int64_t>((info.shape)[1])};
    auto slice_stride = std::vector<int64_t>(1, 1);
    y = new aclnnTensorDescriptor(toAclDataType(info.atype), slice_shape, slice_stride);
    x = new aclnnTensorDescriptor(toAclDataType(info.atype), slice_shape, slice_stride);
    w = new aclnnTensorDescriptor(w_desc);

    // Get AclTensor
    aclTensor *ty = y->tensor;
    aclTensor *tx = x->tensor;
    aclTensor *tw = w->tensor;
    // Set rstdDesc
    // See: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha002/apiref/appdevgapi/context/aclnnRmsNorm.md
    // rstdTensor cannot set nullptr in aclnn
    auto rstd_shape = std::vector<int64_t>(1, 1);
    auto rstd_strides = std::vector<int64_t>(1, 1);
    rstd = new aclnnTensorDescriptor(toAclDataType(INFINI_DTYPE_F32), rstd_shape, rstd_strides);
    aclTensor *trstd = rstd->tensor;

    // Get WorkspaceSize and set executor
    CHECK_ACL(aclnnRmsNormGetWorkspaceSize(tx, tw, static_cast<double>(epsilon), ty, trstd, &workspace_size, &executor));
    aclSetAclOpExecutorRepeatable(executor);

    auto handle_ascend = reinterpret_cast<device::ascend::Handle *>(handle);
    size_t all_workspace_size = workspace_size + rstd->numel() * aclDataTypeSize(rstd->dataType);
    *desc_ptr = new Descriptor(
        new Opaque{executor, y, x, w, rstd, workspace_size},
        std::move(info),
        all_workspace_size,
        handle_ascend->device, handle_ascend->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *y, const void *x, const void *w,
    void *stream) const {

    if (workspace_size < workspaceSize()) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    auto tw = _opaque->w->tensor;
    auto tx = _opaque->x->tensor;
    auto ty = _opaque->y->tensor;
    auto trstd = _opaque->rstd->tensor;

    void *rstdPtr = (void *)((uint8_t *)workspace + _opaque->workspaceSize);

    auto unit = infiniSizeOf(_info.atype);
    AclSetTensorAddr(_opaque->executor, 1, tw, (void *)w);
    AclSetTensorAddr(_opaque->executor, 3, trstd, rstdPtr);
    for (size_t i = 0; i < (_info.shape)[0]; ++i) {
        AclSetTensorAddr(_opaque->executor, 0, tx, ((char *)x) + i * (_info.x_strides)[0] * unit);
        AclSetTensorAddr(_opaque->executor, 2, ty, ((char *)y) + i * (_info.y_strides)[0] * unit);
        CHECK_ACL(aclnnRmsNorm(workspace, _opaque->workspaceSize, _opaque->executor, stream));
    }
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::rms_norm::ascend
