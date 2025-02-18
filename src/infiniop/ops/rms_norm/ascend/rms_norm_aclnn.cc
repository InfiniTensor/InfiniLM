#include "rms_norm_aclnn.h"
#include "../../../devices/ascend/common_ascend.h"
#include <aclnnop/aclnn_cast.h>
#include <aclnnop/aclnn_rms_norm.h>

namespace op::rms_norm::ascend {

struct Descriptor::Opaque {
    mutable aclOpExecutor *executor;
    mutable aclOpExecutor *castExecutor;
    aclnnTensorDescriptor_t y;
    aclnnTensorDescriptor_t x;
    aclnnTensorDescriptor_t w;
    aclnnTensorDescriptor_t rstd;
    aclnnTensorDescriptor_t cast;
    size_t workspaceSize;
    size_t castWorkspaceSize;

    ~Opaque() {
        delete y;
        delete x;
        delete w;
        delete rstd;
        delete cast;
        aclDestroyAclOpExecutor(executor);
        aclDestroyAclOpExecutor(castExecutor);
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
    RMSNormInfo info;
    auto handle_ascend = reinterpret_cast<device::ascend::Handle *>(handle);
    CHECK_STATUS(createRMSNormInfo(&info, y_desc, x_desc, w_desc, epsilon));

    size_t workspace_size, cast_workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclOpExecutor *castExecutor = nullptr;
    aclnnTensorDescriptor_t y = nullptr;
    aclnnTensorDescriptor_t x = nullptr;
    aclnnTensorDescriptor_t w = nullptr;
    aclnnTensorDescriptor_t rstd = nullptr;
    aclnnTensorDescriptor_t cast = nullptr;

    std::vector<int64_t> slice_shape = {1, static_cast<int64_t>((info.shape)[1])};
    auto slice_stride = std::vector<int64_t>(2, 1);
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
    auto rstd_shape = std::vector<int64_t>(2, 1);
    auto rstd_strides = std::vector<int64_t>(2, 1);
    rstd = new aclnnTensorDescriptor(toAclDataType(INFINI_DTYPE_F32), rstd_shape, rstd_strides);
    aclTensor *trstd = rstd->tensor;

    if (w->dataType != x->dataType) {
        cast = new aclnnTensorDescriptor(x->dataType, w->shape, w->strides);
    }

    // Get WorkspaceSize and set executor

    CHECK_ACL(aclnnRmsNormGetWorkspaceSize(tx, cast == nullptr ? tw : cast->tensor, static_cast<double>(epsilon), ty, trstd, &workspace_size, &executor));
    aclSetAclOpExecutorRepeatable(executor);
    if (cast) {
        aclTensor *tc = cast->tensor;
        CHECK_ACL(aclnnCastGetWorkspaceSize(tw, cast->dataType, tc, &cast_workspace_size, &castExecutor));
        aclSetAclOpExecutorRepeatable(castExecutor);
    }

    size_t allWorkspaceSize = workspace_size + cast_workspace_size + rstd->size() * aclDataTypeSize(rstd->dataType);
    allWorkspaceSize = allWorkspaceSize + (cast == nullptr ? 0 : cast->size() * aclDataTypeSize(cast->dataType));
    *desc_ptr = new Descriptor(new Opaque{executor, castExecutor, y, x, w, rstd, cast, workspace_size, cast_workspace_size}, info, allWorkspaceSize, handle_ascend->device, handle_ascend->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *workspace, size_t workspace_size, void *y,
                                     const void *x, const void *w, void *stream) {
    if (workspace_size < workspaceSize()) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    auto tw = _opaque->w->tensor;
    auto tx = _opaque->x->tensor;
    auto ty = _opaque->y->tensor;
    auto trstd = _opaque->rstd->tensor;

    void *rstdPtr = (void *)((uint8_t *)workspace + _opaque->workspaceSize);
    void *castPtr = nullptr;
    if (_opaque->cast) {
        auto tcast = _opaque->cast->tensor;
        castPtr = (void *)((float *)rstdPtr + _opaque->rstd->size());
        AclSetTensorAddr(_opaque->castExecutor, 0, tw, (void *)w);
        AclSetTensorAddr(_opaque->castExecutor, 1, tcast, castPtr);
        CHECK_ACL(aclnnCast(nullptr, _opaque->castWorkspaceSize, _opaque->castExecutor, stream));
    }
    auto unit = infiniSizeOf(_info.atype);
    for (size_t i = 0; i < (_info.shape)[0]; ++i) {
        AclSetTensorAddr(_opaque->executor, 0, tx, ((char *)x) + i * (_info.x_strides)[0] * unit);
        if (_opaque->cast) {
            AclSetTensorAddr(_opaque->executor, 1, _opaque->cast->tensor, castPtr);
        } else {
            AclSetTensorAddr(_opaque->executor, 1, tw, (void *)w);
        }
        AclSetTensorAddr(_opaque->executor, 2, ty, ((char *)y) + i * (_info.y_strides)[0] * unit);
        AclSetTensorAddr(_opaque->executor, 3, trstd, rstdPtr);
        CHECK_ACL(aclnnRmsNorm(workspace, _opaque->workspaceSize, _opaque->executor, stream));
    }
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::rms_norm::ascend
