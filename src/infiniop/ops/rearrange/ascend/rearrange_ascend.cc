#include "rearrange_ascend.h"
#include "../../../devices/ascend/common_ascend.h"
#include <aclnnop/aclnn_copy.h>

namespace op::rearrange::ascend {

struct Descriptor::Opaque {
    aclDataType dt;
    std::vector<int64_t> shape;
    std::vector<int64_t> dst_strides;
    std::vector<int64_t> src_strides;
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

    std::vector<int64_t> shape_(ndim);
    std::vector<int64_t> dst_strides_(ndim);
    std::vector<int64_t> src_strides_(ndim);
    for (size_t i = 0; i < ndim; i++) {
        shape_[i] = static_cast<int64_t>(shape[i]);
        dst_strides_[i] = static_cast<int64_t>(dst_strides[i]);
        src_strides_[i] = static_cast<int64_t>(src_strides[i]);
    }

    *desc_ptr = new Descriptor(
        result.take(),
        new Opaque{
            toAclDataType(dtype),
            shape_,
            dst_strides_,
            src_strides_},
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *y,
    const void *x,
    void *stream) const {

    auto y_ = aclnnTensorDescriptor(_opaque->dt, _opaque->shape, _opaque->dst_strides, y);
    auto x_ = aclnnTensorDescriptor(_opaque->dt, _opaque->shape, _opaque->src_strides, (void *)x);

    auto ty = y_.tensor;
    auto tx = x_.tensor;
    size_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    void *workspace = nullptr;
    CHECK_ACL(aclnnInplaceCopyGetWorkspaceSize(ty, tx, &workspace_size, &executor));
    if (workspace_size != 0) {
        CHECK_ACL(aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    CHECK_ACL(aclnnInplaceCopy(workspace, workspace_size, executor, stream));
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::rearrange::ascend
