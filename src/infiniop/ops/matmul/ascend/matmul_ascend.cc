#include "matmul_ascend.h"
#include "../../../devices/ascend/ascend_handle.h"
#include "../../../devices/ascend/tensor_aclnn.h"
#include <acl/acl_base.h>
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_matmul.h>
#include <aclnnop/level2/aclnn_gemm.h>

namespace op::matmul::ascend {

struct Descriptor::Opaque {
    mutable aclOpExecutor *executor;
    aclnnTensorDescriptor_t c, a, b;
    // cubeMathType
    // see doc:
    // https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha002/apiref/appdevgapi/context/aclnnBatchMatMul.md
    int8_t mt;

    ~Opaque() {
        delete c;
        delete a;
        delete b;
        aclDestroyAclOpExecutor(executor);
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
    auto handle = reinterpret_cast<infiniopAscendHandle_t>(handle_);
    auto dtype = c_desc->dtype();

    if (dtype != INFINI_DTYPE_F16 && dtype != INFINI_DTYPE_F32) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    infiniStatus_t status;
    auto info = MatmulInfo(c_desc, a_desc, b_desc, &status, MatrixLayout::ROW_MAJOR);
    if (status != INFINI_STATUS_SUCCESS) {
        return status;
    }

    auto c = new aclnnTensorDescriptor(),
         a = new aclnnTensorDescriptor(),
         b = new aclnnTensorDescriptor();

    // Treat A, B, C as 2D matrix, reuse aclnnTensorDescriptor for batched
    // operation
    CHECK_STATUS(c->setDescriptor(
        toAclDataType(c_desc->dtype()),
        {static_cast<int64_t>(info.c_matrix.rows),
         static_cast<int64_t>(info.c_matrix.cols)},
        {info.c_matrix.row_stride, info.c_matrix.col_stride}));
    CHECK_STATUS(a->setDescriptor(
        toAclDataType(a_desc->dtype()),
        {static_cast<int64_t>(info.a_matrix.rows),
         static_cast<int64_t>(info.a_matrix.cols)},
        {info.a_matrix.row_stride, info.a_matrix.col_stride}));
    CHECK_STATUS(b->setDescriptor(
        toAclDataType(b_desc->dtype()),
        {static_cast<int64_t>(info.b_matrix.rows),
         static_cast<int64_t>(info.b_matrix.cols)},
        {info.b_matrix.row_stride, info.b_matrix.col_stride}));

    CHECK_STATUS(c->createTensor());
    CHECK_STATUS(a->createTensor());
    CHECK_STATUS(b->createTensor());

    auto tc = c->t,
         ta = a->t,
         tb = b->t;
    aclOpExecutor *executor;
    size_t workspace_size;
    // aclnnGemm support C = alpha * A @ B + beta * C
    // see
    // https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha003/apiref/aolapi/context/aclnnGemm.md
    // use alpha = 0.5, beta = 0.5 temporarily

    int8_t mt = 1;
    CHECK_ACL(aclnnGemmGetWorkspaceSize(ta, tb, tc, .5, .5, 0, 0, tc, mt, &workspace_size, &executor));
    aclSetAclOpExecutorRepeatable(executor);

    *desc_ptr = new Descriptor(
        dtype, info, workspace_size,
        new Opaque{
            executor,
            c,
            a,
            b,
            mt,
        },
        handle->device, handle->device_id);
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

    auto tc = _opaque->c->t,
         ta = _opaque->a->t,
         tb = _opaque->b->t;

    size_t workspace_size;
    CHECK_ACL(aclnnGemmGetWorkspaceSize(
        ta, tb, tc, alpha, beta, 0, 0, tc, _opaque->mt,
        &workspace_size, &(_opaque->executor)));
    if (workspaceSize_ < workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    aclSetAclOpExecutorRepeatable(_opaque->executor);

    auto unit = infiniSizeOf(_dtype);
    for (size_t i = 0; i < _info.batch; ++i) {
        AclSetTensorAddr(_opaque->executor, 0, ta, ((char *)a) + i * _info.a_matrix.stride * unit);
        AclSetTensorAddr(_opaque->executor, 1, tb, ((char *)b) + i * _info.b_matrix.stride * unit);
        AclSetTensorAddr(_opaque->executor, 2, tc, ((char *)c) + i * _info.c_matrix.stride * unit);
        AclSetTensorAddr(_opaque->executor, 3, tc, ((char *)c) + i * _info.c_matrix.stride * unit);
        CHECK_ACL(aclnnGemm(workspace, workspace_size, _opaque->executor, stream));
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::matmul::ascend
