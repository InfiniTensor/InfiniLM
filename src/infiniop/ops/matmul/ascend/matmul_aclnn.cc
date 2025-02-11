#include "matmul_aclnn.h"

MatmulAclnnDescriptor::MatmulAclnnDescriptor(Device _device) {
    device = _device;
    device_id = 0;
    executor = nullptr;
    info = nullptr;
    cDesc = new aclnnTensorDescriptor();
    aDesc = new aclnnTensorDescriptor();
    bDesc = new aclnnTensorDescriptor();
    alpha = 1.0;
    beta = 0;
    mt = 1;
    workspaceSize = 0;
}

infiniopStatus_t aclnnCreateMatmulDescriptor(AscendHandle_t handle,
                                             MatmulAclnnDescriptor_t *desc_ptr,
                                             infiniopTensorDescriptor_t c_desc,
                                             float alpha,
                                             infiniopTensorDescriptor_t a_desc,
                                             infiniopTensorDescriptor_t b_desc,
                                             float beta,
                                             int8_t mt) {
    DT dtype = c_desc->dt;
    if (dtype != F16 && dtype != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    *desc_ptr = new MatmulAclnnDescriptor(handle->device);
    (*desc_ptr)->device_id = handle->device_id;
    (*desc_ptr)->dtype = dtype;
    (*desc_ptr)->mt = mt;
    (*desc_ptr)->alpha = alpha;
    (*desc_ptr)->beta = beta;
    infiniopStatus_t *status = new infiniopStatus_t{STATUS_EXECUTION_FAILED};
    auto info = new MatmulInfo(c_desc, a_desc, b_desc, status, false);
    if (*status != STATUS_SUCCESS) {
        return *status;
    }
    (*desc_ptr)->info = info;

    auto &cDesc = (*desc_ptr)->cDesc;
    auto &aDesc = (*desc_ptr)->aDesc;
    auto &bDesc = (*desc_ptr)->bDesc;

    // Treat A, B, C as 2D matrix, reuse aclnnTensorDescriptor for batched operation
    CHECK_STATUS(cDesc->setDescriptor(toAclDataType(c_desc->dt), {info->c_matrix.rows, info->c_matrix.cols}, {info->c_matrix.row_stride, info->c_matrix.col_stride}), STATUS_SUCCESS);
    CHECK_STATUS(aDesc->setDescriptor(toAclDataType(a_desc->dt), {info->a_matrix.rows, info->a_matrix.cols}, {info->a_matrix.row_stride, info->a_matrix.col_stride}), STATUS_SUCCESS);
    CHECK_STATUS(bDesc->setDescriptor(toAclDataType(b_desc->dt), {info->b_matrix.rows, info->b_matrix.cols}, {info->b_matrix.row_stride, info->b_matrix.col_stride}), STATUS_SUCCESS);

    CHECK_STATUS(cDesc->createTensor(), STATUS_SUCCESS);
    CHECK_STATUS(aDesc->createTensor(), STATUS_SUCCESS);
    CHECK_STATUS(bDesc->createTensor(), STATUS_SUCCESS);


    auto &workspaceSize = (*desc_ptr)->workspaceSize;
    auto &executor = (*desc_ptr)->executor;

    aclTensor *tc = cDesc->t;
    aclTensor *ta = aDesc->t;
    aclTensor *tb = bDesc->t;

    aclnnStatus ret;


    int64_t transA = 0;
    int64_t transB = 0;
    // aclnnGemm support C = alpha * A @ B + beta * C
    // see https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha003/apiref/aolapi/context/aclnnGemm.md
    ret = aclnnGemmGetWorkspaceSize(ta, tb, tc, (*desc_ptr)->alpha, (*desc_ptr)->beta, transA, transB, tc,
                                    (*desc_ptr)->mt, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnGemmGetWorkspaceSize failed. ERROR: %d\n", ret);
              return STATUS_EXECUTION_FAILED);
    aclSetAclOpExecutorRepeatable(executor);

    return STATUS_SUCCESS;
}

infiniopStatus_t aclnnGetMatmulWorkspaceSize(MatmulAclnnDescriptor_t desc,
                                             uint64_t *size) {
    *size = desc->workspaceSize;
    return STATUS_SUCCESS;
}

infiniopStatus_t aclnnMatmul(MatmulAclnnDescriptor_t desc,
                             void *workspace,
                             uint64_t workspace_size,
                             void *c,
                             void const *a,
                             void const *b,
                             void *stream) {
    auto &cDesc = desc->cDesc;
    auto &aDesc = desc->aDesc;
    auto &bDesc = desc->bDesc;

    aclTensor *tc = cDesc->t;
    aclTensor *ta = aDesc->t;
    aclTensor *tb = bDesc->t;

    auto batch = desc->info->batch;

    auto &executor = desc->executor;
    auto &workspaceSize = desc->workspaceSize;

    // Set runing on handle device
    aclrtSetDevice(desc->device_id);

    for (int i = 0; i < batch; i++) {
        AclSetTensorAddr(executor, 0, ta, (char *) (a) + i * desc->info->a_matrix.stride * desc->dtype.size);
        AclSetTensorAddr(executor, 1, tb, (char *) (b) + i * desc->info->b_matrix.stride * desc->dtype.size);
        AclSetTensorAddr(executor, 2, tc, (char *) (c) + i * desc->info->c_matrix.stride * desc->dtype.size);
        AclSetTensorAddr(executor, 3, tc, (char *) (c) + i * desc->info->c_matrix.stride * desc->dtype.size);
        aclnnStatus ret = aclnnGemm(workspace,
                                    workspaceSize,
                                    executor,
                                    stream);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("aclnnGemm failed. ERROR: %d\n", ret);
                  return STATUS_EXECUTION_FAILED);
    }

    return STATUS_SUCCESS;
}


infiniopStatus_t aclnnDestroyMatmulDescriptor(MatmulAclnnDescriptor_t desc) {
    delete desc->cDesc;
    delete desc->bDesc;
    delete desc->aDesc;
    delete desc->info;
    aclDestroyAclOpExecutor(desc->executor);
    delete desc;

    return STATUS_SUCCESS;
}
