#include "matmul_ascend.h"
#include "../../../devices/ascend/tensor_aclnn.h"
#include "../../utils.h"
#include <acl/acl_base.h>
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_matmul.h>
#include <aclnnop/level2/aclnn_gemm.h>

namespace matmul::ascend {

struct Descriptor::Opaque {
    mutable aclOpExecutor *executor;
    aclnnTensorDescriptor_t cDesc, aDesc, bDesc;
    // cubeMathType
    // see doc:
    // https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha002/apiref/appdevgapi/context/aclnnBatchMatMul.md
    int8_t mt;

    ~Opaque() {
        delete cDesc;
        delete aDesc;
        delete bDesc;
        aclDestroyAclOpExecutor(executor);
    }
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniopStatus_t Descriptor::create(
    infiniopAscendHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc) {
    infiniDtype_t dtype = c_desc->dtype;

    if (dtype != INFINI_DTYPE_F16 && dtype != INFINI_DTYPE_F32) {
        return INFINIOP_STATUS_BAD_TENSOR_DTYPE;
    }

    infiniopStatus_t status;
    auto info = MatmulInfo(c_desc, a_desc, b_desc, &status, MatrixLayout::ROW_MAJOR);
    if (status != INFINIOP_STATUS_SUCCESS) {
        return status;
    }

    auto cDesc = new aclnnTensorDescriptor(),
         aDesc = new aclnnTensorDescriptor(),
         bDesc = new aclnnTensorDescriptor();

    // Treat A, B, C as 2D matrix, reuse aclnnTensorDescriptor for batched
    // operation
    CHECK_STATUS(cDesc->setDescriptor(
                     toAclDataType(c_desc->dtype),
                     {static_cast<int64_t>(info.c_matrix.rows),
                      static_cast<int64_t>(info.c_matrix.cols)},
                     {info.c_matrix.row_stride, info.c_matrix.col_stride}),
                 INFINIOP_STATUS_SUCCESS);
    CHECK_STATUS(aDesc->setDescriptor(
                     toAclDataType(a_desc->dtype),
                     {static_cast<int64_t>(info.a_matrix.rows),
                      static_cast<int64_t>(info.a_matrix.cols)},
                     {info.a_matrix.row_stride, info.a_matrix.col_stride}),
                 INFINIOP_STATUS_SUCCESS);
    CHECK_STATUS(bDesc->setDescriptor(
                     toAclDataType(b_desc->dtype),
                     {static_cast<int64_t>(info.b_matrix.rows),
                      static_cast<int64_t>(info.b_matrix.cols)},
                     {info.b_matrix.row_stride, info.b_matrix.col_stride}),
                 INFINIOP_STATUS_SUCCESS);

    CHECK_STATUS(cDesc->createTensor(), INFINIOP_STATUS_SUCCESS);
    CHECK_STATUS(aDesc->createTensor(), INFINIOP_STATUS_SUCCESS);
    CHECK_STATUS(bDesc->createTensor(), INFINIOP_STATUS_SUCCESS);

    auto tc = cDesc->t,
         ta = aDesc->t,
         tb = bDesc->t;
    aclOpExecutor *executor;
    size_t workspaceSize;
    // aclnnGemm support C = alpha * A @ B + beta * C
    // see
    // https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha003/apiref/aolapi/context/aclnnGemm.md
    // use alpha = 0.5, beta = 0.5 temporarily

    int8_t mt = 1;
    auto ret = aclnnGemmGetWorkspaceSize(ta, tb, tc, .5, .5, 0, 0, tc, mt, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnGemmGetWorkspaceSize failed. ERROR: %d\n", ret);
              return INFINIOP_STATUS_INTERNAL_ERROR);
    aclSetAclOpExecutorRepeatable(executor);

    *desc_ptr = new Descriptor(
        dtype, info, workspaceSize,
        new Opaque{
            executor,
            cDesc,
            aDesc,
            bDesc,
            mt,
        },
        handle->device, handle->device_id);
    return INFINIOP_STATUS_SUCCESS;
}

infiniopStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspaceSize_,
    void *c,
    float beta,
    void const *a,
    void const *b,
    float alpha,
    void *stream) const {

    auto tc = _opaque->cDesc->t,
         ta = _opaque->aDesc->t,
         tb = _opaque->bDesc->t;

    size_t workspaceSize;
    auto ret = aclnnGemmGetWorkspaceSize(
        ta, tb, tc, alpha, beta, 0, 0, tc, _opaque->mt,
        &workspaceSize, &(_opaque->executor));
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnGemmGetWorkspaceSize failed. ERROR: %d\n", ret);
              return INFINIOP_STATUS_INTERNAL_ERROR);
    if (workspaceSize_ < workspaceSize) {
        return INFINIOP_STATUS_INSUFFICIENT_WORKSPACE;
    }
    aclSetAclOpExecutorRepeatable(_opaque->executor);

    for (size_t i = 0; i < info.batch; ++i) {
        AclSetTensorAddr(_opaque->executor, 0, ta, ((char *)a) + i * info.a_matrix.stride * infiniSizeof(dtype));
        AclSetTensorAddr(_opaque->executor, 1, tb, ((char *)b) + i * info.b_matrix.stride * infiniSizeof(dtype));
        AclSetTensorAddr(_opaque->executor, 2, tc, ((char *)c) + i * info.c_matrix.stride * infiniSizeof(dtype));
        AclSetTensorAddr(_opaque->executor, 3, tc, ((char *)c) + i * info.c_matrix.stride * infiniSizeof(dtype));
        ret = aclnnGemm(workspace, workspaceSize, _opaque->executor, stream);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("aclnnGemm failed. ERROR: %d\n", ret);
                  return INFINIOP_STATUS_INTERNAL_ERROR);
    }

    return INFINIOP_STATUS_SUCCESS;
}

} // namespace matmul::ascend
