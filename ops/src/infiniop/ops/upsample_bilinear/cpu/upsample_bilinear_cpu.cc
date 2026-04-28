#include "upsample_bilinear_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <vector>

#include "../../../../utils/custom_types.h"

namespace op::upsample_bilinear::cpu {

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
    infiniopTensorDescriptor_t input_desc,
    int align_corners) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);

    // 创建 Info 对象
    auto result = UpsampleBilinearInfo::create(output_desc, input_desc, align_corners);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        new Opaque(),
        result.take(),
        0,
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

// 辅助函数：计算插值权重和索引
struct BilinearParam {
    int64_t idx0;
    int64_t idx1;
    float w0;
    float w1;
};

// 预计算维度的索引和权重
std::vector<BilinearParam> pre_compute_indices_and_weights(
    size_t out_size,
    size_t in_size,
    bool align_corners) {

    std::vector<BilinearParam> params(out_size);

    float scale;
    if (align_corners) {
        scale = (out_size > 1) ? static_cast<float>(in_size - 1) / (out_size - 1) : 0.0f;
    } else {
        scale = static_cast<float>(in_size) / out_size;
    }

    for (size_t i = 0; i < out_size; ++i) {
        float real_idx;
        if (align_corners) {
            real_idx = i * scale;
        } else {
            real_idx = (i + 0.5f) * scale - 0.5f;
            if (real_idx < 0) {
                real_idx = 0; // 防止越界
            }
        }

        int64_t idx0 = static_cast<int64_t>(real_idx);
        int64_t idx1 = idx0 + 1;

        if (idx1 >= static_cast<int64_t>(in_size)) {
            idx1 = in_size - 1;
        }

        float w1 = real_idx - idx0;
        float w0 = 1.0f - w1;

        params[i] = {idx0, idx1, w0, w1};
    }
    return params;
}

template <typename T>
void calculate_cpu_impl(
    const UpsampleBilinearInfo &info,
    void *output,
    const void *input) {

    // 获取形状信息
    size_t N = info.n();
    size_t C = info.c();
    size_t in_h = info.h_in();
    size_t in_w = info.w_in();
    size_t out_h = info.h_out();
    size_t out_w = info.w_out();
    bool align_corners = info.align_corners();

    auto out_ptr = reinterpret_cast<T *>(output);
    auto in_ptr = reinterpret_cast<const T *>(input);

    // 预计算 H 和 W 维度的插值参数
    auto h_params = pre_compute_indices_and_weights(out_h, in_h, align_corners);
    auto w_params = pre_compute_indices_and_weights(out_w, in_w, align_corners);

    size_t n_c = N * C; // 合并 Batch 和 Channel 维度进行并行

#pragma omp parallel for schedule(static)
    for (ptrdiff_t nc = 0; nc < (ptrdiff_t)n_c; ++nc) {
        // 当前 channel 的输入输出起始指针
        const T *src_base = in_ptr + nc * in_h * in_w;
        T *dst_base = out_ptr + nc * out_h * out_w;

        for (size_t h = 0; h < out_h; ++h) {
            const auto &hp = h_params[h];
            // 缓存行指针，避免内层循环重复计算乘法
            const T *src_row0 = src_base + hp.idx0 * in_w;
            const T *src_row1 = src_base + hp.idx1 * in_w;

            for (size_t w = 0; w < out_w; ++w) {
                const auto &wp = w_params[w];

                // 获取四个采样点的值
                float val00 = utils::cast<float>(src_row0[wp.idx0]);
                float val01 = utils::cast<float>(src_row0[wp.idx1]);
                float val10 = utils::cast<float>(src_row1[wp.idx0]);
                float val11 = utils::cast<float>(src_row1[wp.idx1]);

                // 双线性插值计算
                // interpolation = (val00 * w0 + val01 * w1) * h_w0 + (val10 * w0 + val11 * w1) * h_w1
                float val_h0 = val00 * wp.w0 + val01 * wp.w1;
                float val_h1 = val10 * wp.w0 + val11 * wp.w1;
                float result = val_h0 * hp.w0 + val_h1 * hp.w1;

                dst_base[h * out_w + w] = utils::cast<T>(result);
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
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::upsample_bilinear::cpu
