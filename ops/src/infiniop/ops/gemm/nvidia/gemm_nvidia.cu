#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "gemm_nvidia.cuh"

#include <cstdlib>

#if defined(ENABLE_HYGON_API)
#include "gemv_hygon.cuh"
#endif

namespace op::gemm::nvidia {


struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
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
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto dtype = c_desc->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    auto result = MatmulInfo::create(c_desc, a_desc, b_desc, MatrixLayout::COL_MAJOR);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        dtype, result.take(), 0,
        new Opaque{handle->internal()},
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *c,
    float beta,
    const void *a,
    const void *b,
    float alpha,
    void *stream) const {

    cudaDataType a_type, b_type, c_type;
#if defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API)
    cudaDataType compute_type;
#else
    cublasComputeType_t compute_type;
#endif

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        a_type = b_type = c_type = CUDA_R_16F;
#if defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API)
        compute_type = CUDA_R_32F;
#else
        compute_type = CUBLAS_COMPUTE_32F;
#endif
        break;
    case INFINI_DTYPE_BF16:
        a_type = b_type = c_type = CUDA_R_16BF;
#if defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API)
        compute_type = CUDA_R_32F;
#else
        compute_type = CUBLAS_COMPUTE_32F;
#endif
        break;
    case INFINI_DTYPE_F32:
        a_type = b_type = c_type = CUDA_R_32F;
#if defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API)
        compute_type = CUDA_R_32F;
#else
        compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
#endif
        break;

    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    if (_info.is_transed) {
        std::swap(a, b);
    }

    auto op_a = _info.a_matrix.row_stride == 1 ? CUBLAS_OP_N : CUBLAS_OP_T;
    auto op_b = _info.b_matrix.row_stride == 1 ? CUBLAS_OP_N : CUBLAS_OP_T;

#if defined(ENABLE_HYGON_API)
    // Fast path: BF16 GEMV when N=batch=1 (bs=1 decode hot path).
    // PyTorch-style weights: M×K row-major, lda=K, op_a=T. Inputs/outputs
    // are contiguous K- and M-vectors. The DTK hipBLAS heuristic picks
    // MT16x16x32 here which runs at ~10 GFLOPS (~0.02% peak); a purpose-
    // built GEMV reaches HBM-bandwidth-bound throughput. Verified +44%
    // decode tok/s at bs=1 (39.7 → 57.1).
    // Override with `INFINIOPS_DISABLE_GEMV_HYGON=1` to fall back to cuBLAS.
    static const bool kDisableGemv =
        std::getenv("INFINIOPS_DISABLE_GEMV_HYGON") != nullptr;
    if (!kDisableGemv
        && _dtype == INFINI_DTYPE_BF16
        && _info.n == 1 && _info.batch == 1
        && op_a == CUBLAS_OP_T && op_b == CUBLAS_OP_N
        && _info.a_matrix.ld() == (ptrdiff_t)_info.k
        && _info.b_matrix.ld() == (ptrdiff_t)_info.k
        && _info.c_matrix.ld() == (ptrdiff_t)_info.m) {
        cudaError_t err = launch_gemv_bf16(
            reinterpret_cast<const __nv_bfloat16 *>(a),
            reinterpret_cast<const __nv_bfloat16 *>(b),
            reinterpret_cast<__nv_bfloat16 *>(c),
            (int)_info.k, (int)_info.m,
            alpha, beta,
            (cudaStream_t)stream);
        if (err != cudaSuccess) {
            return INFINI_STATUS_INTERNAL_ERROR;
        }
        return INFINI_STATUS_SUCCESS;
    }
#endif

    CHECK_STATUS(_opaque->internal->useCublas(
        (cudaStream_t)stream,
        [&](cublasHandle_t handle) {
            CHECK_CUBLAS(
                cublasGemmStridedBatchedEx(
                    handle,
                    op_a,
                    op_b,
                    static_cast<int>(_info.m),
                    static_cast<int>(_info.n),
                    static_cast<int>(_info.k),
                    &alpha,
                    a,
                    a_type,
                    static_cast<int>(_info.a_matrix.ld()),
                    _info.a_matrix.stride,
                    b,
                    b_type,
                    static_cast<int>(_info.b_matrix.ld()),
                    _info.b_matrix.stride,
                    &beta,
                    c,
                    c_type,
                    static_cast<int>(_info.c_matrix.ld()),
                    _info.c_matrix.stride,
                    static_cast<int>(_info.batch),
                    compute_type,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            return INFINI_STATUS_SUCCESS;
        }));
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::gemm::nvidia
