#include "matmul_cnnl.h"
#include "../../../devices/bang/common_bang.h"
#include "../../utils.h"
#include "matmul_cnnl_api.h"

infiniopStatus_t bangCreateMatmulDescriptor(
    infiniopBangHandle_t handle, infiniopMatmulBangDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t c_desc, infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc) {
    infiniopStatus_t status;
    auto info = MatmulInfo(c_desc, a_desc, b_desc, &status, false);
    if (status != INFINIOP_STATUS_SUCCESS) {
        return status;
    }
    cnnlTensorDescriptor_t aDesc, bDesc, cDesc;
    cnnlCreateTensorDescriptor(&aDesc);
    cnnlCreateTensorDescriptor(&bDesc);
    cnnlCreateTensorDescriptor(&cDesc);

    setMatrixTensorEx(aDesc, info.a_matrix, a_desc->dtype);
    setMatrixTensorEx(bDesc, info.b_matrix, b_desc->dtype);
    setMatrixTensorEx(cDesc, info.c_matrix, c_desc->dtype);

    cnnlMatMulDescriptor_t opDesc;
    cnnlMatMulAlgo_t algo;
    cnnlMatMulHeuristicResult_t algoResult;
    cnnlMatMulDescCreate(&opDesc);
    cnnlMatMulAlgoCreate(&algo);
    cnnlCreateMatMulHeuristicResult(&algoResult);
    int32_t use_stride = true;
    cnnlSetMatMulDescAttr(opDesc, CNNL_MATMUL_USE_STRIDE, &use_stride,
                          sizeof(int32_t));
    int count = 0;
    use_cnnl(handle->cnnl_handle_pool, [&](cnnlHandle_t _handle) {
        cnnlGetBatchMatMulAlgoHeuristic(_handle, opDesc, aDesc, bDesc, cDesc,
                                        NULL, 1, &algoResult, &count);
    });

    size_t workspace_size;
    cnnlGetBatchMatMulHeuristicResult(algoResult, algo, &workspace_size);
    *desc_ptr = new InfiniopMatmulBangDescriptor{handle->device,
                                                 handle->device_id,
                                                 info,
                                                 c_desc->dtype,
                                                 handle->cnnl_handle_pool,
                                                 aDesc,
                                                 bDesc,
                                                 cDesc,
                                                 opDesc,
                                                 algo,
                                                 algoResult,
                                                 workspace_size};

    return INFINIOP_STATUS_SUCCESS;
}
infiniopStatus_t bangGetMatmulWorkspaceSize(infiniopMatmulBangDescriptor_t desc,
                                            size_t *size) {
    *size = desc->workspace_size;
    return INFINIOP_STATUS_SUCCESS;
}

infiniopStatus_t
bangDestroyMatmulDescriptor(infiniopMatmulBangDescriptor_t desc) {
    desc->cnnl_handle_pool = nullptr;
    cnnlDestroyTensorDescriptor(desc->aDesc);
    cnnlDestroyTensorDescriptor(desc->bDesc);
    cnnlDestroyTensorDescriptor(desc->cDesc);
    cnnlMatMulDescDestroy(desc->opDesc);
    cnnlMatMulAlgoDestroy(desc->algo);
    cnnlDestroyMatMulHeuristicResult(desc->algoResult);
    delete desc;
    return INFINIOP_STATUS_SUCCESS;
}

void bangMatmulCnnl(infiniopMatmulBangDescriptor_t desc, void *workspace, void *c,
                 float beta, void const *a, void const *b, float alpha,
                 void *stream) {
    auto info = desc->info;
    if (info.is_transed) {
        std::swap(a, b);
    }

    use_cnnl(desc->cnnl_handle_pool, (cnrtQueue_t)stream, [&](cnnlHandle_t handle) {
        cnnlBatchMatMulBCast_v2(handle, desc->opDesc, desc->algo, &alpha,
                                desc->aDesc, a, desc->bDesc, b, &beta,
                                desc->cDesc, c, workspace,
                                desc->workspace_size);
    });
}
infiniopStatus_t bangMatmul(infiniopMatmulBangDescriptor_t desc,
                            void *workspace, size_t workspace_size, void *c,
                            void const *a, void const *b, float alpha,
                            float beta, void *stream) {
    if (desc->dtype == INFINI_DTYPE_F16 || desc->dtype == INFINI_DTYPE_F32) {
        bangMatmulCnnl(desc, workspace, c, beta, a, b, alpha, stream);
        cnrtQueueSync((cnrtQueue_t)stream);
        return INFINIOP_STATUS_SUCCESS;
    }
    return INFINIOP_STATUS_BAD_TENSOR_DTYPE;
}
