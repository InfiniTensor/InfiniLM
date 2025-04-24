#include "gemm_ascend.h"
#include "../../../devices/ascend/common_ascend.h"
#include <aclnnop/aclnn_matmul.h>
#include <aclnnop/level2/aclnn_gemm.h>

namespace op::gemm::ascend {

struct Descriptor::Opaque {
    aclnnTensorDescriptor_t c, a, b;
    // cubeMathType
    // see doc:
    // https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha002/apiref/appdevgapi/context/aclnnBatchMatMul.md
    int8_t mt;

    ~Opaque() {
        delete c;
        delete a;
        delete b;
    }
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc) {
    auto handle = reinterpret_cast<device::ascend::Handle *>(handle_);
    auto dtype = c_desc->dtype();

    if (dtype != INFINI_DTYPE_F16 && dtype != INFINI_DTYPE_F32) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    auto result = MatmulInfo::create(c_desc, a_desc, b_desc, MatrixLayout::ROW_MAJOR);
    CHECK_RESULT(result);
    auto info = result.take();

    auto c = new aclnnTensorDescriptor(toAclDataType(c_desc->dtype()),
                                       {static_cast<int64_t>(info.m), static_cast<int64_t>(info.n)},
                                       {info.c_matrix.row_stride, info.c_matrix.col_stride});
    auto a = new aclnnTensorDescriptor(toAclDataType(a_desc->dtype()),
                                       {static_cast<int64_t>(info.a_matrix.rows), static_cast<int64_t>(info.a_matrix.cols)},
                                       {info.a_matrix.row_stride, info.a_matrix.col_stride});
    auto b = new aclnnTensorDescriptor(toAclDataType(b_desc->dtype()),
                                       {static_cast<int64_t>(info.b_matrix.rows), static_cast<int64_t>(info.b_matrix.cols)},
                                       {info.b_matrix.row_stride, info.b_matrix.col_stride});

    auto tc = c->tensor,
         ta = a->tensor,
         tb = b->tensor;

    aclOpExecutor *executor = nullptr;
    size_t workspace_size = 0;
    // aclnnGemm support C = alpha * A @ B + beta * C
    // see
    // https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha003/apiref/aolapi/context/aclnnGemm.md
    // use alpha = 0.5, beta = 0.5 temporarily

    int8_t mt = 1;
    CHECK_ACL(aclnnGemmGetWorkspaceSize(ta, tb, tc, .5, .5, 0, 0, tc, mt, &workspace_size, &executor));

    *desc_ptr = new Descriptor(
        dtype, info, workspace_size,
        new Opaque{
            c,
            a,
            b,
            mt,
        },
        handle->device, handle->device_id);

    aclDestroyAclOpExecutor(executor);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspaceSize_,
    void *c,
    float beta,
    const void *a,
    const void *b,
    float alpha,
    void *stream) const {

    auto tc = _opaque->c->tensor,
         ta = _opaque->a->tensor,
         tb = _opaque->b->tensor;

    size_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    CHECK_ACL(aclnnGemmGetWorkspaceSize(
        ta, tb, tc, alpha, beta, 0, 0, tc, _opaque->mt,
        &workspace_size, &executor));
    if (workspaceSize_ < workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    CHECK_ACL(aclSetAclOpExecutorRepeatable(executor));

    auto unit = infiniSizeOf(_dtype);
    for (size_t i = 0; i < _info.batch; ++i) {
        AclSetTensorAddr(executor, 0, ta, ((char *)a) + i * _info.a_matrix.stride * unit);
        AclSetTensorAddr(executor, 1, tb, ((char *)b) + i * _info.b_matrix.stride * unit);
        AclSetTensorAddr(executor, 2, tc, ((char *)c) + i * _info.c_matrix.stride * unit);
        AclSetTensorAddr(executor, 3, tc, ((char *)c) + i * _info.c_matrix.stride * unit);
        CHECK_ACL(aclnnGemm(workspace, workspace_size, executor, stream));
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::gemm::ascend
