#include "matmul_cnnl.h"
#include "../../../devices/bang/bang_handle.h"
#include "../../../devices/bang/common_bang.h"
#include "../../utils.h"
#include "cnrt.h"
infiniopStatus_t bangCreateMatmulDescriptor(BangHandle_t handle,
                                            MatmulBangDescriptor_t *desc_ptr,
                                            infiniopTensorDescriptor_t c_desc,
                                            float alpha,
                                            infiniopTensorDescriptor_t a_desc,
                                            infiniopTensorDescriptor_t b_desc,
                                            float beta) {
    infiniopStatus_t *status = new infiniopStatus_t{STATUS_EXECUTION_FAILED};
    auto info = MatmulInfo(c_desc, a_desc, b_desc, status, false);
    if (*status != STATUS_SUCCESS) {
        return *status;
    }
    cnnlTensorDescriptor_t aDesc, bDesc, cDesc;
    cnnlCreateTensorDescriptor(&aDesc);
    cnnlCreateTensorDescriptor(&bDesc);
    cnnlCreateTensorDescriptor(&cDesc);

    setMatrixTensorEx(aDesc, info.a_matrix);
    setMatrixTensorEx(bDesc, info.b_matrix);
    setMatrixTensorEx(cDesc, info.c_matrix);

    cnnlMatMulDescriptor_t opDesc;
    cnnlMatMulAlgo_t algo;
    cnnlMatMulHeuristicResult_t algoResult;
    cnnlMatMulDescCreate(&opDesc);
    cnnlMatMulAlgoCreate(&algo);
    cnnlCreateMatMulHeuristicResult(&algoResult);
    int32_t use_stride = true;
    cnnlSetMatMulDescAttr(opDesc, CNNL_MATMUL_USE_STRIDE, &use_stride,
                          sizeof(int32_t));
    *desc_ptr = new MatmulBangDescriptor{
        handle->device,
        handle->device_id,
        info,
        alpha,
        beta,
        c_desc->dt,
        handle->cnnl_handles,
        aDesc,
        bDesc,
        cDesc,
        opDesc,
        algo,
        algoResult};
    return STATUS_SUCCESS;
}
infiniopStatus_t bangGetMatmulWorkspaceSize(MatmulBangDescriptor_t desc, uint64_t *size) {
    *size = 0;
    return STATUS_SUCCESS;
}

infiniopStatus_t bangDestroyMatmulDescriptor(MatmulBangDescriptor_t desc) {
    desc->cnnl_handles = nullptr;
    cnnlDestroyTensorDescriptor(desc->aDesc);
    cnnlDestroyTensorDescriptor(desc->bDesc);
    cnnlDestroyTensorDescriptor(desc->cDesc);
    cnnlMatMulDescDestroy(desc->opDesc);
    cnnlMatMulAlgoDestroy(desc->algo);
    cnnlDestroyMatMulHeuristicResult(desc->algoResult);
    delete desc;
    return STATUS_SUCCESS;
}

void matmul_cnnl_f16(MatmulBangDescriptor_t desc, void *workspace, void *c, float beta, void const *a, void const *b, float alpha, void *stream) {
    auto info = desc->info;
    if (info.is_transed) {
        std::swap(a, b);
    }

    use_cnnl(desc->cnnl_handles, desc->device_id, (cnrtQueue_t) stream,
             [&](cnnlHandle_t handle) {
                 int count = 0;
                 cnnlGetBatchMatMulAlgoHeuristic(handle, desc->opDesc, desc->aDesc,
                                                 desc->bDesc, desc->cDesc,
                                                 NULL, 1, &desc->algoResult, &count);
                 size_t wsSize;
                 cnnlGetBatchMatMulHeuristicResult(desc->algoResult, desc->algo, &wsSize);
                 cnrtMalloc(&workspace, wsSize);
                 cnnlBatchMatMulBCast_v2(handle, desc->opDesc, desc->algo,
                                         &alpha, desc->aDesc, a,
                                         desc->bDesc, b,
                                         &beta, desc->cDesc, c,
                                         workspace, wsSize);
             });
}
infiniopStatus_t bangMatmul(MatmulBangDescriptor_t desc, void *workspace, uint64_t workspace_size, void *c, void const *a, void const *b, void *stream) {
    if (cnrtSetDevice(desc->device_id) != cnrtSuccess) {
        return STATUS_BAD_DEVICE;
    }
    float alpha = desc->alpha;
    float beta = desc->beta;
    if (dtype_eq(desc->dtype, F16)) {
        matmul_cnnl_f16(desc, workspace, c, beta, a, b, alpha, stream);
        cnrtQueueSync((cnrtQueue_t)stream);
        return STATUS_SUCCESS;
    }
    return STATUS_BAD_TENSOR_DTYPE;
}
