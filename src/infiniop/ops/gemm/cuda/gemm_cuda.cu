#include "../../../devices/cuda/cuda_handle.cuh"
#include "gemm_cuda.cuh"

namespace op::gemm::cuda {

struct Descriptor::Opaque {
    std::shared_ptr<device::cuda::Handle::Internal> internal;
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
    auto handle = reinterpret_cast<device::cuda::nvidia::Handle *>(handle_);
    auto dtype = c_desc->dtype();

    if (dtype != INFINI_DTYPE_F16 && dtype != INFINI_DTYPE_F32) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

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
    cublasComputeType_t compute_type;

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        a_type = b_type = c_type = CUDA_R_16F;
        compute_type = CUBLAS_COMPUTE_32F;
        break;

    case INFINI_DTYPE_F32:
        a_type = b_type = c_type = CUDA_R_32F;
#ifdef ENABLE_SUGON_CUDA_API
        compute_type = CUBLAS_COMPUTE_32F;
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

} // namespace op::gemm::cuda
