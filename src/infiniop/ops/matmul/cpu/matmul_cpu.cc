#include "matmul_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../devices/cpu/cpu_handle.h"

namespace matmul::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc) {
    auto handle = reinterpret_cast<infiniopCpuHandle_t>(handle_);
    auto dtype = c_desc->dtype();

    if (dtype != INFINI_DTYPE_F16 && dtype != INFINI_DTYPE_F32) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    infiniStatus_t status;
    auto info = MatmulInfo(c_desc, a_desc, b_desc, &status, MatrixLayout::COL_MAJOR);
    if (status != INFINI_STATUS_SUCCESS) {
        return status;
    }

    *desc_ptr = new Descriptor(
        dtype, info, 0,
        nullptr,
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata>
void calculate(
    const MatmulInfo &info,
    void *c,
    float beta,
    const void *a,
    const void *b,
    float alpha) {
    if (info.is_transed) {
        std::swap(a, b);
    }

    for (size_t i = 0; i < info.batch; ++i) {
        for (size_t m_ = 0; m_ < info.m; ++m_) {
            for (size_t n_ = 0; n_ < info.n; ++n_) {
                auto c_ = reinterpret_cast<Tdata *>(c) + i * info.c_matrix.stride + m_ * info.c_matrix.row_stride + n_ * info.c_matrix.col_stride;
                float sum = 0;
                for (size_t k_ = 0; k_ < info.k; ++k_) {
                    auto a_ = reinterpret_cast<const Tdata *>(a) + i * info.a_matrix.stride + m_ * info.a_matrix.row_stride + k_ * info.a_matrix.col_stride;
                    auto b_ = reinterpret_cast<const Tdata *>(b) + i * info.b_matrix.stride + n_ * info.b_matrix.col_stride + k_ * info.b_matrix.row_stride;
                    if constexpr (std::is_same<Tdata, uint16_t>::value) {
                        sum += f16_to_f32(*a_) * f16_to_f32(*b_);
                    } else {
                        sum += *a_ * (*b_);
                    }
                }
                if constexpr (std::is_same<Tdata, uint16_t>::value) {
                    if (beta == 0) {
                        *c_ = f32_to_f16(alpha * sum);
                    } else {
                        *c_ = f32_to_f16(beta * f16_to_f32(*c_) + alpha * sum);
                    }
                } else {
                    *c_ = beta * (*c_) + alpha * sum;
                }
            }
        }
    }
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

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        cpu::calculate<uint16_t>(_info, c, beta, a, b, alpha);
        return INFINI_STATUS_SUCCESS;

    case INFINI_DTYPE_F32:
        cpu::calculate<float>(_info, c, beta, a, b, alpha);
        return INFINI_STATUS_SUCCESS;

    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace matmul::cpu
