#include "addbmm_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../handle.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stddef.h>
#include <vector>

namespace op::addbmm::cpu {

Descriptor::~Descriptor() = default;

// ==================================================================
// 辅助函数：通用 stride 寻址
// ==================================================================

// 计算 2D 张量的偏移量
inline size_t offset_2d(size_t r, size_t c, const int64_t *strides) {
    return r * strides[0] + c * strides[1];
}

// 计算 3D 张量的偏移量
inline size_t offset_3d(size_t b, size_t r, size_t c, const int64_t *strides) {
    return b * strides[0] + r * strides[1] + c * strides[2];
}

// ==================================================================
// 核心 Kernel 实现
// ==================================================================

/**
 * @brief Addbmm 核心 CPU 计算函数 (支持任意 Stride)
 */
template <typename Tdata>
void calculate_impl(
    const AddbmmInfo &info,
    void *output,
    const void *input,
    const void *batch1,
    const void *batch2) {

    // [变更 1] 使用 Getter 获取维度
    size_t b_dim = info.b();
    size_t n = info.n();
    size_t m = info.m();
    size_t p = info.p();

    float alpha = info.alpha();
    float beta = info.beta();

    // 指针转换
    Tdata *out_ptr = reinterpret_cast<Tdata *>(output);
    const Tdata *inp_ptr = reinterpret_cast<const Tdata *>(input);
    const Tdata *b1_ptr = reinterpret_cast<const Tdata *>(batch1);
    const Tdata *b2_ptr = reinterpret_cast<const Tdata *>(batch2);

    const int64_t *out_strides = info.out_strides().data();
    const int64_t *in_strides = info.in_strides().data();
    const int64_t *b1_strides = info.b1_strides().data();
    const int64_t *b2_strides = info.b2_strides().data();

    // 1. 初始化 output = beta * input
    for (size_t i = 0; i < n; ++i) {
        for (size_t k = 0; k < p; ++k) {
            size_t out_idx = offset_2d(i, k, out_strides);
            size_t in_idx = offset_2d(i, k, in_strides);

            float val_in = (beta != 0.0f) ? utils::cast<float>(inp_ptr[in_idx]) : 0.0f;

            if (beta == 0.0f && alpha == 0.0f) {
                out_ptr[out_idx] = utils::cast<Tdata>(0.0f);
            } else {
                out_ptr[out_idx] = utils::cast<Tdata>(val_in * beta);
            }
        }
    }

    // 2. 累加矩阵乘法: out += alpha * sum(b1 @ b2)
    for (size_t b = 0; b < b_dim; ++b) {     // Batch
        for (size_t i = 0; i < n; ++i) {     // Row
            for (size_t k = 0; k < p; ++k) { // Col

                float dot_product = 0.0f;

                // 内部点积 (Inner dimension m)
                for (size_t j = 0; j < m; ++j) {
                    size_t b1_idx = offset_3d(b, i, j, b1_strides);
                    size_t b2_idx = offset_3d(b, j, k, b2_strides);

                    float v1 = utils::cast<float>(b1_ptr[b1_idx]);
                    float v2 = utils::cast<float>(b2_ptr[b2_idx]);

                    dot_product += v1 * v2;
                }

                // 累加到 Output
                size_t out_idx = offset_2d(i, k, out_strides);
                float current_val = utils::cast<float>(out_ptr[out_idx]);
                out_ptr[out_idx] = utils::cast<Tdata>(current_val + alpha * dot_product);
            }
        }
    }
}

// ==================================================================
// Descriptor 接口实现
// ==================================================================

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec,
    float alpha,
    float beta) {

    if (input_desc_vec.size() != 3) {
        return INFINI_STATUS_BAD_PARAM;
    }

    infiniopTensorDescriptor_t in_desc = input_desc_vec[0];
    infiniopTensorDescriptor_t batch1_desc = input_desc_vec[1];
    infiniopTensorDescriptor_t batch2_desc = input_desc_vec[2];

    auto dtype = out_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16, INFINI_DTYPE_F64);

    // 创建 Info 对象
    auto result = AddbmmInfo::create(out_desc, in_desc, batch1_desc, batch2_desc, alpha, beta);
    CHECK_RESULT(result);

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);

    *desc_ptr = new Descriptor(
        nullptr,
        result.take(),
        0,
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const {

    if (inputs.size() != 3) {
        return INFINI_STATUS_BAD_PARAM;
    }

    const void *input = inputs[0];
    const void *batch1 = inputs[1];
    const void *batch2 = inputs[2];

    auto dtype = _info.dtype();

    switch (dtype) {
    case INFINI_DTYPE_F16:
        calculate_impl<fp16_t>(_info, output, input, batch1, batch2);
        return INFINI_STATUS_SUCCESS;

    case INFINI_DTYPE_BF16:
        calculate_impl<bf16_t>(_info, output, input, batch1, batch2);
        return INFINI_STATUS_SUCCESS;

    case INFINI_DTYPE_F32:
        calculate_impl<float>(_info, output, input, batch1, batch2);
        return INFINI_STATUS_SUCCESS;

    case INFINI_DTYPE_F64:
        calculate_impl<double>(_info, output, input, batch1, batch2);
        return INFINI_STATUS_SUCCESS;

    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::addbmm::cpu
