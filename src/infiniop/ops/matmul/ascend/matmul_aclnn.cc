#include "matmul_aclnn.h"

InfiniopMatmulAclnnDescriptor::InfiniopMatmulAclnnDescriptor(
    infiniDevice_t _device) {
    device = _device;
    device_id = 0;
    executor = nullptr;
    info = nullptr;
    cDesc = new aclnnTensorDescriptor();
    aDesc = new aclnnTensorDescriptor();
    bDesc = new aclnnTensorDescriptor();
    mt = 1;
    workspaceSize = 0;
}

infiniopStatus_t aclnnCreateMatmulDescriptor(infiniopAscendHandle_t handle,
                                             MatmulAclnnDescriptor_t *desc_ptr,
                                             infiniopTensorDescriptor_t c_desc,
                                             infiniopTensorDescriptor_t a_desc,
                                             infiniopTensorDescriptor_t b_desc,
                                             int8_t mt) {
    infiniDtype_t dtype = c_desc->dtype;
    if (dtype != INFINI_DTYPE_F16 && dtype != INFINI_DTYPE_F32) {
        return INFINIOP_STATUS_BAD_TENSOR_DTYPE;
    }

    *desc_ptr = new InfiniopMatmulAclnnDescriptor(handle->device);
    (*desc_ptr)->device_id = handle->device_id;
    (*desc_ptr)->dtype = dtype;
    (*desc_ptr)->mt = mt;
    infiniopStatus_t status;
    auto info = new MatmulInfo(c_desc, a_desc, b_desc, &status, false);
    if (status != INFINIOP_STATUS_SUCCESS) {
        return status;
    }
    (*desc_ptr)->info = info;

    auto &cDesc = (*desc_ptr)->cDesc;
    auto &aDesc = (*desc_ptr)->aDesc;
    auto &bDesc = (*desc_ptr)->bDesc;

    // Treat A, B, C as 2D matrix, reuse aclnnTensorDescriptor for batched
    // operation
    CHECK_STATUS(cDesc->setDescriptor(
                     toAclDataType(c_desc->dtype),
                     {static_cast<int64_t>(info->c_matrix.rows),
                      static_cast<int64_t>(info->c_matrix.cols)},
                     {info->c_matrix.row_stride, info->c_matrix.col_stride}),
                 INFINIOP_STATUS_SUCCESS);
    CHECK_STATUS(aDesc->setDescriptor(
                     toAclDataType(a_desc->dtype),
                     {static_cast<int64_t>(info->a_matrix.rows),
                      static_cast<int64_t>(info->a_matrix.cols)},
                     {info->a_matrix.row_stride, info->a_matrix.col_stride}),
                 INFINIOP_STATUS_SUCCESS);
    CHECK_STATUS(bDesc->setDescriptor(
                     toAclDataType(b_desc->dtype),
                     {static_cast<int64_t>(info->b_matrix.rows),
                      static_cast<int64_t>(info->b_matrix.cols)},
                     {info->b_matrix.row_stride, info->b_matrix.col_stride}),
                 INFINIOP_STATUS_SUCCESS);

    CHECK_STATUS(cDesc->createTensor(), INFINIOP_STATUS_SUCCESS);
    CHECK_STATUS(aDesc->createTensor(), INFINIOP_STATUS_SUCCESS);
    CHECK_STATUS(bDesc->createTensor(), INFINIOP_STATUS_SUCCESS);

    auto &workspaceSize = (*desc_ptr)->workspaceSize;
    auto &executor = (*desc_ptr)->executor;

    aclTensor *tc = cDesc->t;
    aclTensor *ta = aDesc->t;
    aclTensor *tb = bDesc->t;

    aclnnStatus ret;

    int64_t transA = 0;
    int64_t transB = 0;
    // aclnnGemm support C = alpha * A @ B + beta * C
    // see
    // https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha003/apiref/aolapi/context/aclnnGemm.md
    // use alpha = 0.5, beta = 0.5 temporarily
    ret = aclnnGemmGetWorkspaceSize(ta, tb, tc, 0.5f, 0.5f, transA, transB, tc,
                                    (*desc_ptr)->mt, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnGemmGetWorkspaceSize failed. ERROR: %d\n", ret);
              return INFINIOP_STATUS_INTERNAL_ERROR);
    aclSetAclOpExecutorRepeatable(executor);

    return INFINIOP_STATUS_SUCCESS;
}

infiniopStatus_t aclnnGetMatmulWorkspaceSize(MatmulAclnnDescriptor_t desc,
                                             size_t *size) {
    *size = desc->workspaceSize;
    return INFINIOP_STATUS_SUCCESS;
}

infiniopStatus_t aclnnMatmul(MatmulAclnnDescriptor_t desc, void *workspace,
                             size_t workspace_size, void *c, void const *a,
                             void const *b, float alpha, float beta,
                             void *stream) {
    auto &cDesc = desc->cDesc;
    auto &aDesc = desc->aDesc;
    auto &bDesc = desc->bDesc;

    aclTensor *tc = cDesc->t;
    aclTensor *ta = aDesc->t;
    aclTensor *tb = bDesc->t;

    auto batch = desc->info->batch;

    size_t workspaceSize;
    aclnnStatus ret;
    ret = aclnnGemmGetWorkspaceSize(ta, tb, tc, alpha, beta, 0, 0, tc, desc->mt,
                                    &workspaceSize, &(desc->executor));
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnGemmGetWorkspaceSize failed. ERROR: %d\n", ret);
              return INFINIOP_STATUS_INTERNAL_ERROR);
    if (workspace_size < workspaceSize) {
        return INFINIOP_STATUS_INSUFFICIENT_WORKSPACE;
    }
    aclSetAclOpExecutorRepeatable(desc->executor);

    for (size_t i = 0; i < batch; i++) {
        AclSetTensorAddr(desc->executor, 0, ta,
                         (char *)(a) + i * desc->info->a_matrix.stride * infiniSizeof(desc->dtype));
        AclSetTensorAddr(desc->executor, 1, tb,
                         (char *)(b) + i * desc->info->b_matrix.stride * infiniSizeof(desc->dtype));
        AclSetTensorAddr(desc->executor, 2, tc,
                         (char *)(c) + i * desc->info->c_matrix.stride * infiniSizeof(desc->dtype));
        AclSetTensorAddr(desc->executor, 3, tc,
                         (char *)(c) + i * desc->info->c_matrix.stride * infiniSizeof(desc->dtype));
        ret = aclnnGemm(workspace, workspaceSize, desc->executor, stream);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("aclnnGemm failed. ERROR: %d\n", ret);
                  return INFINIOP_STATUS_INTERNAL_ERROR);
    }

    return INFINIOP_STATUS_SUCCESS;
}

infiniopStatus_t aclnnDestroyMatmulDescriptor(MatmulAclnnDescriptor_t desc) {
    delete desc->cDesc;
    delete desc->bDesc;
    delete desc->aDesc;
    delete desc->info;
    aclDestroyAclOpExecutor(desc->executor);
    delete desc;

    return INFINIOP_STATUS_SUCCESS;
}
