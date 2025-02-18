#include "../../utils.h"
#include "./matmul_cuda.cuh"

infiniopStatus_t cudaCreateMatmulDescriptor(infiniopCudaHandle_t handle,
                                            infiniopMatmulCudaDescriptor_t *desc_ptr,
                                            infiniopTensorDescriptor_t c_desc,
                                            infiniopTensorDescriptor_t a_desc,
                                            infiniopTensorDescriptor_t b_desc) {
    infiniDtype_t dtype = c_desc->dtype;

    if (dtype != INFINI_DTYPE_F16 && dtype != INFINI_DTYPE_F32) {
        return INFINIOP_STATUS_BAD_TENSOR_DTYPE;
    }

    infiniopStatus_t status;
    auto info = MatmulInfo(c_desc, a_desc, b_desc, &status);
    if (status != INFINIOP_STATUS_SUCCESS) {
        return status;
    }

    *desc_ptr = new InfiniopMatmulCudaDescriptor{
        handle->device,
        dtype,
        handle->device_id,
        info,
        handle->cublas_handle_pool};
    return INFINIOP_STATUS_SUCCESS;
}

infiniopStatus_t cudaGetMatmulWorkspaceSize(infiniopMatmulCudaDescriptor_t desc, size_t *size) {
    *size = 0;
    return INFINIOP_STATUS_SUCCESS;
}

infiniopStatus_t cudaDestroyMatmulDescriptor(infiniopMatmulCudaDescriptor_t desc) {
    desc->cublas_handle_pool = nullptr;
    delete desc;
    return INFINIOP_STATUS_SUCCESS;
}
