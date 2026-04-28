#include "affine_grid_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include <algorithm>
#include <cmath>
#include <omp.h>

namespace op::affine_grid::cpu {

template <typename T>
inline float to_float(T val) {
    if constexpr (std::is_same<T, fp16_t>::value || std::is_same<T, bf16_t>::value) {
        return utils::cast<float>(val);
    } else {
        return static_cast<float>(val);
    }
}

template <typename T>
inline T from_float(float val) {
    if constexpr (std::is_same<T, fp16_t>::value || std::is_same<T, bf16_t>::value) {
        return utils::cast<T>(val);
    } else {
        return static_cast<T>(val);
    }
}

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t in_desc,
    bool align_corners) { // 接收 align_corners

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto dtype = out_desc->dtype();

    // 1. 检查数据类型
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16, INFINI_DTYPE_F64);

    // 2. 创建 Info 对象 (传递 align_corners)
    auto result = AffineGridInfo::create(out_desc, in_desc, align_corners);
    CHECK_RESULT(result);

    // 3. 创建 Descriptor
    *desc_ptr = new Descriptor(
        nullptr,       // Opaque*
        result.take(), // Info
        0,             // Workspace Size (AffineGrid CPU 不需要额外 workspace)
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata>
void calculate_cpu_impl(
    const AffineGridInfo &info,
    void *output,
    const void *input) {

    size_t batch = info.batch();
    size_t H = info.height();
    size_t W = info.width();
    bool align_corners = info.align_corners();

    auto out_ptr = reinterpret_cast<Tdata *>(output);
    auto in_ptr = reinterpret_cast<const Tdata *>(input);

    // 并行化处理 Batch
#pragma omp parallel for
    for (ptrdiff_t n = 0; n < (ptrdiff_t)batch; ++n) {

        const Tdata *theta_n = in_ptr + n * 6;

        // 提取仿射矩阵参数并转为 float
        float r00 = to_float(theta_n[0]);
        float r01 = to_float(theta_n[1]);
        float tx = to_float(theta_n[2]);
        float r10 = to_float(theta_n[3]);
        float r11 = to_float(theta_n[4]);
        float ty = to_float(theta_n[5]);

        // 遍历空间维度
        for (size_t h = 0; h < H; ++h) {
            for (size_t w = 0; w < W; ++w) {
                // 1. 计算归一化坐标 (-1 到 1)
                float x_norm, y_norm;

                if (align_corners) {
                    x_norm = (W > 1) ? (2.0f * w) / (W - 1.0f) - 1.0f : 0.0f;
                    y_norm = (H > 1) ? (2.0f * h) / (H - 1.0f) - 1.0f : 0.0f;
                } else {
                    x_norm = (2.0f * w + 1.0f) / W - 1.0f;
                    y_norm = (2.0f * h + 1.0f) / H - 1.0f;
                }

                // 2. 应用仿射变换
                float grid_x = r00 * x_norm + r01 * y_norm + tx;
                float grid_y = r10 * x_norm + r11 * y_norm + ty;
                size_t offset = (n * H * W + h * W + w) * 2;

                out_ptr[offset + 0] = from_float<Tdata>(grid_x);
                out_ptr[offset + 1] = from_float<Tdata>(grid_y);
            }
        }
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream) const {

    // 从 Info 中获取 dtype
    auto dtype = _info.dtype();

    switch (dtype) {
    case INFINI_DTYPE_F16:
        cpu::calculate_cpu_impl<fp16_t>(_info, output, input);
        return INFINI_STATUS_SUCCESS;

    case INFINI_DTYPE_BF16:
        cpu::calculate_cpu_impl<bf16_t>(_info, output, input);
        return INFINI_STATUS_SUCCESS;

    case INFINI_DTYPE_F32:
        cpu::calculate_cpu_impl<float>(_info, output, input);
        return INFINI_STATUS_SUCCESS;

    case INFINI_DTYPE_F64:
        cpu::calculate_cpu_impl<double>(_info, output, input);
        return INFINI_STATUS_SUCCESS;

    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::affine_grid::cpu
