#include "../../utils.h"
#include "matmul_cuda.cuh"

namespace matmul::cuda {

struct Descriptor::Opaque {
    std::shared_ptr<Pool<cublasHandle_t>> cublas_handle_pool;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniopStatus_t Descriptor::create(
    infiniopCudaHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc) {
    infiniDtype_t dtype = c_desc->dtype;

    if (dtype != INFINI_DTYPE_F16 && dtype != INFINI_DTYPE_F32) {
        return INFINIOP_STATUS_BAD_TENSOR_DTYPE;
    }

    infiniopStatus_t status;
    auto info = MatmulInfo(c_desc, a_desc, b_desc, &status, MatrixLayout::COL_MAJOR);
    if (status != INFINIOP_STATUS_SUCCESS) {
        return status;
    }

    *desc_ptr = new Descriptor(
        dtype, info, 0,
        new Opaque{handle->cublas_handle_pool},
        handle->device, handle->device_id);
    return INFINIOP_STATUS_SUCCESS;
}

template <typename Tdata>
infiniopStatus_t calculate(
    MatmulInfo const &info,
    std::shared_ptr<Pool<cublasHandle_t>> &cublas_handle_pool,
    void *c,
    float beta,
    void const *a,
    void const *b,
    float alpha,
    cudaStream_t stream) {

    if (info.is_transed) {
        std::swap(a, b);
    }

    cudaDataType a_type, b_type, c_type;
    cublasComputeType_t compute_type;
    if constexpr (std::is_same<Tdata, half>::value) {
        a_type = b_type = c_type = CUDA_R_16F;
        compute_type = CUBLAS_COMPUTE_32F;
    } else {
        a_type = b_type = c_type = CUDA_R_32F;
#ifdef ENABLE_SUGON_CUDA_API
        compute_type = CUBLAS_COMPUTE_32F;
#else
        compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
#endif
    }

    auto op_a = info.a_matrix.row_stride == 1 ? CUBLAS_OP_N : CUBLAS_OP_T;
    auto op_b = info.b_matrix.row_stride == 1 ? CUBLAS_OP_N : CUBLAS_OP_T;

    use_cublas(cublas_handle_pool,
               stream,
               [&](cublasHandle_t handle) {
                   cublasGemmStridedBatchedEx(
                       handle,
                       op_a,
                       op_b,
                       static_cast<int>(info.m),
                       static_cast<int>(info.n),
                       static_cast<int>(info.k),
                       &alpha,
                       a,
                       a_type,
                       static_cast<int>(info.a_matrix.ld()),
                       info.a_matrix.stride,
                       b,
                       b_type,
                       static_cast<int>(info.b_matrix.ld()),
                       info.b_matrix.stride,
                       &beta,
                       c,
                       c_type,
                       static_cast<int>(info.c_matrix.ld()),
                       info.c_matrix.stride,
                       static_cast<int>(info.batch),
                       compute_type,
                       CUBLAS_GEMM_DEFAULT_TENSOR_OP);
               });
    return INFINIOP_STATUS_SUCCESS;
}

infiniopStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *c,
    float beta,
    void const *a,
    void const *b,
    float alpha,
    void *stream) const {

    switch (dtype) {
    case INFINI_DTYPE_F16:
        cuda::calculate<uint16_t>(info, _opaque->cublas_handle_pool, c, beta, a, b, alpha, (cudaStream_t)stream);
        return INFINIOP_STATUS_SUCCESS;

    case INFINI_DTYPE_F32:
        cuda::calculate<float>(info, _opaque->cublas_handle_pool, c, beta, a, b, alpha, (cudaStream_t)stream);
        return INFINIOP_STATUS_SUCCESS;

    default:
        return INFINIOP_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace matmul::cuda
