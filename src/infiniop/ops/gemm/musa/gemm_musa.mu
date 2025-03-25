#include "../../../devices/musa/common_musa.h"
#include "../../../devices/musa/musa_handle.h"
#include "gemm_musa.h"

namespace op::gemm::musa {

struct Descriptor::Opaque {
    std::shared_ptr<device::musa::Handle::Internal> internal;
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
    auto handle = reinterpret_cast<device::musa::Handle *>(handle_);
    auto dtype = c_desc->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32);

    infiniStatus_t status;
    auto info = MatmulInfo(c_desc, a_desc, b_desc, &status, MatrixLayout::COL_MAJOR);
    if (status != INFINI_STATUS_SUCCESS) {
        return status;
    }

    *desc_ptr = new Descriptor(
        dtype, info, 0,
        new Opaque{handle->internal()},
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata>
infiniStatus_t calculate(
    const MatmulInfo &info,
    std::shared_ptr<device::musa::Handle::Internal> &_internal,
    void *c,
    float beta,
    const void *a,
    const void *b,
    float alpha,
    void *stream) {

    musaDataType a_type, b_type, c_type;
    mublasComputeType_t compute_type;
    Tdata alpha_, beta_;

    if constexpr (std::is_same<Tdata, half>::value) {
        alpha_ = __float2half(alpha);
        beta_ = __float2half(beta);
        a_type = b_type = c_type = MUSA_R_16F;
        compute_type = MUBLAS_COMPUTE_16F;
    } else {
        alpha_ = alpha;
        beta_ = beta;
        a_type = b_type = c_type = MUSA_R_32F;
        compute_type = MUBLAS_COMPUTE_32F_FAST_TF32;
    }

    if (info.is_transed) {
        std::swap(a, b);
    }

    auto op_a = info.a_matrix.row_stride == 1 ? MUBLAS_OP_N : MUBLAS_OP_T;
    auto op_b = info.b_matrix.row_stride == 1 ? MUBLAS_OP_N : MUBLAS_OP_T;

    CHECK_STATUS(_internal->useMublas(
        (musaStream_t)stream,
        [&](mublasHandle_t handle) {
            CHECK_MUBLAS(
                mublasGemmStridedBatchedEx(
                    handle,
                    op_a,
                    op_b,
                    static_cast<int>(info.m),
                    static_cast<int>(info.n),
                    static_cast<int>(info.k),
                    &alpha_,
                    a,
                    a_type,
                    static_cast<int>(info.a_matrix.ld()),
                    info.a_matrix.stride,
                    b,
                    b_type,
                    static_cast<int>(info.b_matrix.ld()),
                    info.b_matrix.stride,
                    &beta_,
                    c,
                    c_type,
                    static_cast<int>(info.c_matrix.ld()),
                    info.c_matrix.stride,
                    static_cast<int>(info.batch),
                    compute_type,
                    MUBLAS_GEMM_DEFAULT));
            return INFINI_STATUS_SUCCESS;
        }));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *workspace,
                                     size_t workspace_size,
                                     void *c,
                                     float beta,
                                     const void *a,
                                     const void *b,
                                     float alpha,
                                     void *stream) const {
    switch (_dtype) {
        case INFINI_DTYPE_F16:
            return musa::calculate<half>(_info, _opaque->internal, c, beta, a, b, alpha, stream);
        case INFINI_DTYPE_F32:
            return musa::calculate<float>(_info,_opaque->internal, c, beta, a, b, alpha, stream);
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::gemm::musa
