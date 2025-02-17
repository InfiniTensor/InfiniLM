#include "./matmul_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"
#include <cmath>

infiniopStatus_t cpuCreateMatmulDescriptor(
    infiniopCpuHandle_t handle, infiniopMatmulCpuDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t c_desc, infiniopTensorDescriptor_t a_desc,
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

    *desc_ptr = new MatmulCpuDescriptor{INFINI_DEVICE_CPU, dtype, info};

    return INFINIOP_STATUS_SUCCESS;
}

infiniopStatus_t cpuGetMatmulWorkspaceSize(infiniopMatmulCpuDescriptor_t desc,
                                           uint64_t *size) {
    *size = 0;
    return INFINIOP_STATUS_SUCCESS;
}

infiniopStatus_t
cpuDestroyMatmulDescriptor(infiniopMatmulCpuDescriptor_t desc) {
    delete desc;
    return INFINIOP_STATUS_SUCCESS;
}

template <typename Tdata>
infiniopStatus_t cpuCalculateMatmul(infiniopMatmulCpuDescriptor_t desc, void *c,
                                    float beta, void const *a, void const *b,
                                    float alpha) {
    auto info = desc->info;

    if (info.is_transed) {
        std::swap(a, b);
    }

    for (size_t i = 0; i < info.batch; ++i) {
        for (size_t m_ = 0; m_ < info.m; ++m_) {
            for (size_t n_ = 0; n_ < info.n; ++n_) {
                auto c_ = reinterpret_cast<Tdata *>(c) + i * info.c_matrix.stride + m_ * info.c_matrix.row_stride + n_ * info.c_matrix.col_stride;
                float sum = 0;
                for (size_t k_ = 0; k_ < info.k; ++k_) {
                    auto a_ = reinterpret_cast<Tdata const *>(a) + i * info.a_matrix.stride + m_ * info.a_matrix.row_stride + k_ * info.a_matrix.col_stride;
                    auto b_ = reinterpret_cast<Tdata const *>(b) + i * info.b_matrix.stride + n_ * info.b_matrix.col_stride + k_ * info.b_matrix.row_stride;
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
    return INFINIOP_STATUS_SUCCESS;
}

infiniopStatus_t cpuMatmul(infiniopMatmulCpuDescriptor_t desc, void *workspace,
                           uint64_t workspace_size, void *c, void const *a,
                           void const *b, float alpha, float beta) {
    if (desc->dtype == INFINI_DTYPE_F16) {
        return cpuCalculateMatmul<uint16_t>(desc, c, beta, a, b, alpha);
    }
    if (desc->dtype == INFINI_DTYPE_F32) {
        return cpuCalculateMatmul<float>(desc, c, beta, a, b, alpha);
    }
    return INFINIOP_STATUS_BAD_TENSOR_DTYPE;
}
