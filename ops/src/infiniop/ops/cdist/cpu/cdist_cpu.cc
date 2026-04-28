#include "cdist_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include <cmath>

namespace op::cdist::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x1_desc,
    infiniopTensorDescriptor_t x2_desc,
    double p) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto dtype = y_desc->dtype();

    // 1. 类型检查
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    // 2. 解析维度信息 (CdistInfo 逻辑已在之前定义)
    auto result = CdistInfo::create(y_desc, x1_desc, x2_desc);
    CHECK_RESULT(result);

    // 3. 实例化描述符，CPU 版通常不需要 workspace
    *desc_ptr = new Descriptor(
        dtype, result.take(), p, 0,
        nullptr,
        handle->device, handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

/**
 * 核心计算模板：处理不同数据类型
 */
template <typename Tdata>
void calculate_dist(
    const CdistInfo &info,
    void *y,
    const void *x1,
    const void *x2,
    double p) {

    // Flatten loops: batch * m * n for OpenMP parallelization
    const ptrdiff_t total = ptrdiff_t(info.batch) * ptrdiff_t(info.m) * ptrdiff_t(info.n);

#pragma omp parallel for
    for (ptrdiff_t idx = 0; idx < total; ++idx) {
        ptrdiff_t b = idx / (info.m * info.n);
        ptrdiff_t rem = idx % (info.m * info.n);
        ptrdiff_t i = rem / info.n;
        ptrdiff_t j = rem % info.n;

        // output pointer: y[b, i, j]
        auto y_ptr = reinterpret_cast<Tdata *>(y)
                   + b * info.y_matrix.stride
                   + i * info.y_matrix.row_stride
                   + j * info.y_matrix.col_stride;

        // input vectors: x1[b, i, :] and x2[b, j, :]
        auto x1_vec = reinterpret_cast<const Tdata *>(x1)
                    + b * info.x1_matrix.stride
                    + i * info.x1_matrix.row_stride;
        auto x2_vec = reinterpret_cast<const Tdata *>(x2)
                    + b * info.x2_matrix.stride
                    + j * info.x2_matrix.row_stride;

        double dist = 0.0;

        for (size_t k = 0; k < info.d; ++k) {
            float v1 = utils::cast<float>(*(x1_vec + k * info.x1_matrix.col_stride));
            float v2 = utils::cast<float>(*(x2_vec + k * info.x2_matrix.col_stride));
            float diff = std::abs(v1 - v2);

            if (p == 1.0) {
                dist += diff;
            } else if (p == 2.0) {
                dist += diff * diff;
            } else if (std::isinf(p)) {
                dist = std::max(dist, static_cast<double>(diff));
            } else {
                dist += std::pow(static_cast<double>(diff), p);
            }
        }

        // final distance
        if (p == 2.0) {
            dist = std::sqrt(dist);
        } else if (!std::isinf(p) && p != 1.0) {
            dist = std::pow(dist, 1.0 / p);
        }

        *y_ptr = utils::cast<Tdata>(static_cast<float>(dist));
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x1,
    const void *x2,
    void *stream) const {

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        cpu::calculate_dist<fp16_t>(_info, y, x1, x2, _p);
        return INFINI_STATUS_SUCCESS;

    case INFINI_DTYPE_BF16:
        cpu::calculate_dist<bf16_t>(_info, y, x1, x2, _p);
        return INFINI_STATUS_SUCCESS;

    case INFINI_DTYPE_F32:
        cpu::calculate_dist<float>(_info, y, x1, x2, _p);
        return INFINI_STATUS_SUCCESS;

    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::cdist::cpu
