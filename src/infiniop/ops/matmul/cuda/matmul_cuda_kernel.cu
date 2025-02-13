#include "../../utils.h"
#include "./matmul_cuda.cuh"

template<typename Tdata>
infiniopStatus_t cudaMatmulCublas(infiniopMatmulCudaDescriptor_t desc, void *c, float beta, void const *a, void const *b, float alpha, void *stream) {
    auto info = desc->info;

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

    use_cublas(desc->cublas_handles_t, desc->device_id, (cudaStream_t) stream,
               [&](cublasHandle_t handle) { cublasGemmStridedBatchedEx(
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
                                                CUBLAS_GEMM_DEFAULT_TENSOR_OP); });
    return INFINIOP_STATUS_SUCCESS;
}

infiniopStatus_t cudaMatmul(infiniopMatmulCudaDescriptor_t desc,
                            void *workspace,
                            uint64_t workspace_size,
                            void *c,
                            void const *a,
                            void const *b,
                            float alpha,
                            float beta,
                            void *stream) {
    if (desc->dtype == INFINI_DTYPE_F16) {
        return cudaMatmulCublas<half>(desc, c, beta, a, b, alpha, stream);
    }
    if (desc->dtype == INFINI_DTYPE_F32) {
        return cudaMatmulCublas<float>(desc, c, beta, a, b, alpha, stream);
    }
    return INFINIOP_STATUS_BAD_TENSOR_DTYPE;
}
