#include "../../../../devices/moore/moore_common.h"
#include "../../../../devices/moore/moore_handle.h"
#include "gemm_mublas.h"

namespace op::gemm::mublas {

struct Descriptor::Opaque {
    std::shared_ptr<device::moore::Handle::Internal> internal;
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
    auto handle = reinterpret_cast<device::moore::Handle *>(handle_);
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

    musaDataType a_type, b_type, c_type;
    mublasComputeType_t compute_type;

    // MUSA's GEMM operations require that the scalar values alpha and beta have the same data type as the matrices.
    // This ensures correct computation during the muBLAS GEMM operation.
    // Declare half-precision variables to handle F16 types.
    half alpha_h, beta_h;

    // Initialize generic void pointers for alpha and beta.
    // They point to the original float values 
    // It will be used directly when the GEMM operation is performed with F32 data.
    const void *p_alpha = &alpha;
    const void *p_beta = &beta;

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        a_type = b_type = c_type = MUSA_R_16F;
        compute_type = MUBLAS_COMPUTE_16F;

        // Convert alpha/beta to half-precision and update the pointers.
        alpha_h = __float2half(alpha);
        beta_h = __float2half(beta);
        p_alpha = &alpha_h;
        p_beta = &beta_h;

        break;
    case INFINI_DTYPE_BF16:
        a_type = b_type = c_type = MUSA_R_16BF;
        compute_type = MUBLAS_COMPUTE_32F;
        break;
    case INFINI_DTYPE_F32:
        a_type = b_type = c_type = MUSA_R_32F;
        compute_type = MUBLAS_COMPUTE_32F_FAST_TF32;
        break;

    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    if (_info.is_transed) {
        std::swap(a, b);
    }

    auto op_a = _info.a_matrix.row_stride == 1 ? MUBLAS_OP_N : MUBLAS_OP_T;
    auto op_b = _info.b_matrix.row_stride == 1 ? MUBLAS_OP_N : MUBLAS_OP_T;

    CHECK_STATUS(_opaque->internal->useMublas(
        (musaStream_t)stream,
        [&](mublasHandle_t handle) {
            CHECK_MUBLAS(
                mublasGemmStridedBatchedEx(
                    handle,
                    op_a,
                    op_b,
                    static_cast<int>(_info.m),
                    static_cast<int>(_info.n),
                    static_cast<int>(_info.k),
                    p_alpha,
                    a,
                    a_type,
                    static_cast<int>(_info.a_matrix.ld()),
                    _info.a_matrix.stride,
                    b,
                    b_type,
                    static_cast<int>(_info.b_matrix.ld()),
                    _info.b_matrix.stride,
                    p_beta,
                    c,
                    c_type,
                    static_cast<int>(_info.c_matrix.ld()),
                    _info.c_matrix.stride,
                    static_cast<int>(_info.batch),
                    compute_type,
                    MUBLAS_GEMM_DEFAULT));
            return INFINI_STATUS_SUCCESS;
        }));
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::gemm::mublas
