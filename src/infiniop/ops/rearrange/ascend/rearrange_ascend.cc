#include "rearrange_ascend.h"
#include "../../../devices/ascend/common_ascend.h"
#include <aclnnop/aclnn_copy.h>

namespace op::rearrange::ascend {

struct Descriptor::Opaque {
    aclnnTensorDescriptor_t dst;
    aclnnTensorDescriptor_t src;
    void *workspace; // aclnnInplaceCopy workspace
    uint64_t workspace_size;
    ~Opaque() {
        delete dst;
        delete src;

        aclrtFree(workspace);
    }
};

Descriptor::~Descriptor() {
    delete _opaque;
};

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc) {
    auto handle = reinterpret_cast<device::ascend::Handle *>(handle_);
    auto dtype = y_desc->dtype();
    auto ndim = y_desc->ndim();
    auto shape = y_desc->shape();
    CHECK_API_OR(x_desc->dtype(), dtype, return INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_API_OR(x_desc->ndim(), ndim, return INFINI_STATUS_BAD_TENSOR_SHAPE);

    for (size_t i = 0; i < ndim; ++i) {
        CHECK_API_OR(x_desc->shape()[i], shape[i], return INFINI_STATUS_BAD_TENSOR_SHAPE);
    }
    auto dst_strides = y_desc->strides();
    auto src_strides = x_desc->strides();
    auto element_size = infiniSizeOf(dtype);

    auto result = utils::RearrangeMeta::create(shape.data(), dst_strides.data(), src_strides.data(), ndim, element_size);
    CHECK_RESULT(result);

    aclnnTensorDescriptor_t dst = new aclnnTensorDescriptor(y_desc);
    aclnnTensorDescriptor_t src = new aclnnTensorDescriptor(x_desc);

    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    void *workspace = nullptr;
    aclnnInplaceCopyGetWorkspaceSize(dst->tensor, src->tensor,
                                     &workspace_size, &executor);
    if (workspace_size != 0) {
        CHECK_ACL(aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    *desc_ptr = new Descriptor(
        result.take(),
        new Opaque{
            dst,
            src,
            workspace,
            workspace_size},
        handle->device,
        handle->device_id);

    // Delete useless executor
    aclDestroyAclOpExecutor(executor);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *y,
    const void *x,
    void *stream) const {
    auto tdst = _opaque->dst->tensor;
    auto tsrc = _opaque->src->tensor;

    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    AclSetTensorAddr(executor, 0, tdst, y);
    AclSetTensorAddr(executor, 1, tsrc, (void *)x);
    CHECK_ACL(aclnnInplaceCopyGetWorkspaceSize(tdst, tsrc, &workspace_size, &executor));
    // Execute InplaceCopy
    CHECK_ACL(aclnnInplaceCopy(_opaque->workspace, _opaque->workspace_size,
                               executor, stream));

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::rearrange::ascend
