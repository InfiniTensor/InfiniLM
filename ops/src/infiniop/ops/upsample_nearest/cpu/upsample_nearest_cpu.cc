#include "upsample_nearest_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <vector>

#include "../../../../utils/custom_types.h"

namespace op::upsample_nearest::cpu {

struct Descriptor::Opaque {};

Descriptor::~Descriptor() {
    if (_opaque) {
        delete _opaque;
        _opaque = nullptr;
    }
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);

    // 创建 Info 对象
    auto result = UpsampleNearestInfo::create(output_desc, input_desc);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        new Opaque(),
        result.take(),
        0,
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

// 辅助函数：预计算维度的索引
// Nearest 插值只需要知道输出坐标对应的输入整数坐标
std::vector<int64_t> pre_compute_indices(
    size_t out_size,
    size_t in_size) {

    std::vector<int64_t> indices(out_size);

    // 计算缩放因子
    float scale = static_cast<float>(in_size) / out_size;

    for (size_t i = 0; i < out_size; ++i) {
        // Nearest 逻辑：通常向下取整
        // src_idx = floor(dst_idx * scale)
        int64_t idx = static_cast<int64_t>(std::floor(i * scale));

        // 防止越界 (虽理论上不应发生，但为了稳健性)
        if (idx >= static_cast<int64_t>(in_size)) {
            idx = in_size - 1;
        }
        indices[i] = idx;
    }
    return indices;
}

template <typename T>
void calculate_cpu_impl(
    const UpsampleNearestInfo &info,
    void *output,
    const void *input) {

    // 获取形状信息
    size_t N = info.n();
    size_t C = info.c();
    size_t in_h = info.h_in();
    size_t in_w = info.w_in();
    size_t out_h = info.h_out();
    size_t out_w = info.w_out();

    auto out_ptr = reinterpret_cast<T *>(output);
    auto in_ptr = reinterpret_cast<const T *>(input);

    // 预计算 H 和 W 维度的索引映射
    auto h_indices = pre_compute_indices(out_h, in_h);
    auto w_indices = pre_compute_indices(out_w, in_w);

    size_t n_c = N * C; // 合并 Batch 和 Channel 维度进行并行

#pragma omp parallel for schedule(static)
    for (ptrdiff_t nc = 0; nc < (ptrdiff_t)n_c; ++nc) {
        // 当前 channel 的输入输出起始指针
        const T *src_base = in_ptr + nc * in_h * in_w;
        T *dst_base = out_ptr + nc * out_h * out_w;

        for (size_t h = 0; h < out_h; ++h) {
            // 获取当前输出行对应的输入行索引
            int64_t src_h = h_indices[h];
            // 缓存该行的输入指针
            const T *src_row = src_base + src_h * in_w;
            // 缓存该行的输出指针
            T *dst_row = dst_base + h * out_w;

            for (size_t w = 0; w < out_w; ++w) {
                // 获取当前输出列对应的输入列索引
                int64_t src_w = w_indices[w];

                // 直接赋值
                dst_row[w] = src_row[src_w];
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

    auto dtype = _info.dtype();

    switch (dtype) {
    case INFINI_DTYPE_F32:
        cpu::calculate_cpu_impl<float>(_info, output, input);
        break;
    case INFINI_DTYPE_F64:
        cpu::calculate_cpu_impl<double>(_info, output, input);
        break;
    case INFINI_DTYPE_F16:
        cpu::calculate_cpu_impl<fp16_t>(_info, output, input);
        break;
    case INFINI_DTYPE_BF16:
        cpu::calculate_cpu_impl<bf16_t>(_info, output, input);
        break;
    case INFINI_DTYPE_U8:
        cpu::calculate_cpu_impl<uint8_t>(_info, output, input);
        break;
    case INFINI_DTYPE_I8:
        cpu::calculate_cpu_impl<int8_t>(_info, output, input);
        break;
    case INFINI_DTYPE_I16:
        cpu::calculate_cpu_impl<int16_t>(_info, output, input);
        break;
    case INFINI_DTYPE_U16:
        cpu::calculate_cpu_impl<uint16_t>(_info, output, input);
        break;
    case INFINI_DTYPE_I32:
        cpu::calculate_cpu_impl<int32_t>(_info, output, input);
        break;
    case INFINI_DTYPE_U32:
        cpu::calculate_cpu_impl<uint32_t>(_info, output, input);
        break;
    case INFINI_DTYPE_I64:
        cpu::calculate_cpu_impl<int64_t>(_info, output, input);
        break;
    case INFINI_DTYPE_U64:
        cpu::calculate_cpu_impl<uint64_t>(_info, output, input);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::upsample_nearest::cpu
