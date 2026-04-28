#if defined ENABLE_QY_API
#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "dlblas_ext.h"
#include "gptq_qyblas_gemm_nvidia.cuh"

inline cudaDataType_t ScalarTypeToCudaDataType(
    infiniDtype_t scalar_type) {
    switch (scalar_type) {
    case INFINI_DTYPE_U8:
        return CUDA_R_8U;
    case INFINI_DTYPE_I8:
        return CUDA_R_8I;
    case INFINI_DTYPE_I32:
        return CUDA_R_32I;
    case INFINI_DTYPE_F16:
        return CUDA_R_16F;
    case INFINI_DTYPE_F32:
        return CUDA_R_32F;
    case INFINI_DTYPE_F64:
        return CUDA_R_64F;
    case INFINI_DTYPE_I16:
        return CUDA_R_16I;
    case INFINI_DTYPE_I64:
        return CUDA_R_64I;
    case INFINI_DTYPE_BF16:
        return CUDA_R_16BF;
    case INFINI_DTYPE_F8:
        return (cudaDataType_t)CUDA_R_8F_E4M3;
    default:
        fprintf(stderr,
                "Cannot convert ScalarType %d\n",
                (int)scalar_type);
        abort();
    }
}
namespace op::gptq_qyblas_gemm::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle, Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc,
    infiniopTensorDescriptor_t b_scales_desc,
    infiniopTensorDescriptor_t b_zeros_desc) {

    auto info = GptqQyblasGemmInfo::createGptqQyblasGemmInfo(out_desc, a_desc, b_desc, b_scales_desc, b_zeros_desc);

    CHECK_RESULT(info);

    size_t workspace_size = 0;
    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        info.take(), workspace_size, handle->device, handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *workspace,
                                     size_t workspace_size,
                                     void *out,
                                     const void *a,
                                     const void *b,
                                     void *b_scales,
                                     void *b_zeros,
                                     int64_t quant_type,
                                     int64_t bit,
                                     void *stream) const {
    int64_t K = static_cast<int64_t>(_info.K);

    cudaDataType_t computeType_ = (cudaDataType_t)CUDA_R_32F;
    cudaDataType_t kernel_Atype_, kernel_Btype_, kernel_Ctype_, kernel_Stype_, kernel_Ztype_;
    auto dtype = _info.dtype;
    auto weight_dtype = _info.weight_dtype;
    if (_info.transpose_result) {
        std::swap(a, b);
        std::swap(dtype, weight_dtype);
    }
    kernel_Atype_ = ScalarTypeToCudaDataType(dtype);
    kernel_Btype_ = ScalarTypeToCudaDataType(weight_dtype);

    if (quant_type == 0) {
        if (8 == bit) {
            kernel_Atype_ = (cudaDataType_t)CUDA_R_8U;
        }

        if (4 == bit) {
            kernel_Atype_ = (cudaDataType_t)CUDA_R_4U;
            K = K * 2;
        }
    }

    kernel_Ctype_ = ScalarTypeToCudaDataType(_info.out_dtype);
    kernel_Stype_ = ScalarTypeToCudaDataType(_info.scales_dtype);
    kernel_Ztype_ = ScalarTypeToCudaDataType(_info.zeros_dtype);

    float alpha = 1.0f;
    float beta = 0.0f;

    int64_t M = static_cast<int64_t>(_info.M);
    int64_t N = static_cast<int64_t>(_info.N);
    int64_t lda = static_cast<int64_t>(_info.lda);
    int64_t ldb = static_cast<int64_t>(_info.ldb);

    int64_t scales_size_0 = static_cast<int64_t>(_info.scales_size_0);
    int64_t scales_size_1 = static_cast<int64_t>(_info.scales_size_1);

    int64_t result_ld = static_cast<int64_t>(_info.result_ld);

    dlblasExtQuantParametersV2_t extParameters;

    if (quant_type == 0) {
        extParameters.a_group_size_m = M / scales_size_1;
        extParameters.a_group_size_k = K / scales_size_0;
        extParameters.a_zeropoints_type = kernel_Ztype_;
        extParameters.a_zeropoints = b_zeros;
        extParameters.a_scales_type = kernel_Stype_;
        extParameters.a_scales = b_scales;
    } else if (quant_type == 1) {
        extParameters.a_group_size_m = 1;
        extParameters.a_group_size_k = K;
        extParameters.a_zeropoints = nullptr;
        extParameters.a_scales_type = kernel_Stype_;
        extParameters.a_scales = b_scales;

    } else if (quant_type == 2 || quant_type == 3) {
        // calculate block_shape according weight/scales shape
        int block_shape = 128;
        while ((M + block_shape - 1) / block_shape < scales_size_0) {
            block_shape /= 2;
            if (block_shape < 32) {
                fprintf(stderr,
                        "INTERNAL ASSERT FAILED: block_shape >= 32\n"
                        "Invalid fp blockwise linear arguments. Weight: [%d, %d]. Scales: [%d, %d].\n",
                        (int)M, (int)K, (int)scales_size_0, (int)scales_size_1);
                abort();
            }
        }
        if (!((K + block_shape - 1) / block_shape == scales_size_1)) {
            fprintf(stderr,
                    "CHECK FAILED: (K + block_shape - 1) / block_shape == scales_size_1\n");
            abort();
        }
        extParameters.a_group_size_m = block_shape;
        extParameters.a_group_size_k = block_shape;
        extParameters.a_scales_type = kernel_Stype_;
        extParameters.a_zeropoints = nullptr;
        extParameters.a_scales = b_scales;
    }
    bool transpose_mat_1 = _info.transa == 't';
    bool transpose_mat_2 = _info.transb == 't';
    cublasOperation_t transa = transpose_mat_1 ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transb = transpose_mat_2 ? CUBLAS_OP_T : CUBLAS_OP_N;

    if (_info.dtype == INFINI_DTYPE_F16 || _info.dtype == INFINI_DTYPE_BF16) {
        CHECK_STATUS(_opaque->internal->useCublas(
            (cudaStream_t)stream,
            [&](cublasHandle_t handle) {
                CHECK_CUBLAS(
                    dlblasGemmExV2(handle,
                                   transa,
                                   transb,
                                   M,
                                   N,
                                   K,
                                   &alpha,
                                   a,
                                   kernel_Atype_,
                                   lda,
                                   b,
                                   kernel_Btype_,
                                   ldb,
                                   &beta,
                                   out,
                                   kernel_Ctype_,
                                   result_ld,
                                   computeType_,
                                   CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                                   &extParameters));
                return INFINI_STATUS_SUCCESS;
            }));
        return INFINI_STATUS_SUCCESS;
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::gptq_qyblas_gemm::nvidia
#endif
