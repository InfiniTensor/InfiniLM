#include "unfold_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include <algorithm>
#include <cstring>
#include <omp.h>

#include "../../../../utils/custom_types.h"

namespace op::unfold::cpu {

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
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t input_desc,
    const int *kernel_sizes,
    const int *strides,
    const int *paddings,
    const int *dilations) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);

    // Call the static create method from UnfoldInfo
    auto result = UnfoldInfo::infer(out_desc, input_desc, kernel_sizes, strides, paddings, dilations);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        new Opaque(),
        result.take(),
        0, // No workspace needed for CPU
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

// Core Calculation Implementation
template <typename T>
void calculate_cpu_impl(
    const UnfoldInfo &info,
    void *output,
    const void *input) {

    if (info._kernel_sizes.size() < 2) {
        return;
    }

    int64_t batch = info._N;
    int64_t in_c = info._C_in;

    int64_t in_h = info._input_spatial_shape[0];
    int64_t in_w = info._input_spatial_shape[1];

    int64_t out_h = info._output_spatial_shape[0];
    int64_t out_w = info._output_spatial_shape[1];

    int64_t k_h = info._kernel_sizes[0];
    int64_t k_w = info._kernel_sizes[1];
    int64_t stride_h = info._strides[0];
    int64_t stride_w = info._strides[1];
    int64_t pad_h = info._paddings[0];
    int64_t pad_w = info._paddings[1];
    int64_t dil_h = info._dilations[0];
    int64_t dil_w = info._dilations[1];

    auto out_ptr = reinterpret_cast<T *>(output);
    auto in_ptr = reinterpret_cast<const T *>(input);

    int64_t L = info._L;
    int64_t out_c_dim = info._C_out;

    int64_t total_nc = batch * in_c;

#pragma omp parallel for schedule(static)
    for (int64_t idx = 0; idx < total_nc; ++idx) {

        int64_t n = idx / in_c;
        int64_t c = idx % in_c;

        int64_t in_batch_offset = n * in_c * in_h * in_w;
        int64_t out_batch_offset = n * out_c_dim * L;

        for (int64_t kh = 0; kh < k_h; ++kh) {
            for (int64_t kw = 0; kw < k_w; ++kw) {

                int64_t out_c_idx = c * k_h * k_w + kh * k_w + kw;

                for (int64_t oh = 0; oh < out_h; ++oh) {
                    for (int64_t ow = 0; ow < out_w; ++ow) {

                        int64_t h_in = oh * stride_h - pad_h + kh * dil_h;
                        int64_t w_in = ow * stride_w - pad_w + kw * dil_w;

                        int64_t out_idx = out_batch_offset + out_c_idx * L + (oh * out_w + ow);

                        if (h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w) {

                            int64_t in_idx = in_batch_offset + c * in_h * in_w + h_in * in_w + w_in;

                            out_ptr[out_idx] = in_ptr[in_idx];
                        } else {
                            out_ptr[out_idx] = utils::cast<T>(0.0f);
                        }
                    }
                }
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

    auto dtype = _info.dtype_val();

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

} // namespace op::unfold::cpu
