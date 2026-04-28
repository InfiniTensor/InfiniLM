#include "adaptive_avg_pool1d_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include <algorithm>
#include <cmath>

namespace op::adaptive_avg_pool1d::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t in_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto dtype = out_desc->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16, INFINI_DTYPE_F64);
    auto result = AdaptiveAvgPool1dInfo::create(out_desc, in_desc);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        nullptr,       // Opaque*
        result.take(), // Info
        0,             // Workspace Size
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata>
void calculate(
    const AdaptiveAvgPool1dInfo &info,
    void *output,
    const void *input) {

    size_t num_channels = info.num_channels();
    size_t isize = info.input_size();
    size_t osize = info.output_size();

    auto out_ptr = reinterpret_cast<Tdata *>(output);
    auto in_ptr = reinterpret_cast<const Tdata *>(input);

#pragma omp parallel for
    for (ptrdiff_t c = 0; c < (ptrdiff_t)num_channels; ++c) {

        const Tdata *in_c = in_ptr + c * isize;
        Tdata *out_c = out_ptr + c * osize;

        for (size_t i = 0; i < osize; ++i) {

            size_t istart = (i * isize) / osize;
            size_t iend = ((i + 1) * isize + osize - 1) / osize;

            size_t klen = iend - istart;

            float sum = 0.0f;

            for (size_t j = istart; j < iend; ++j) {
                if constexpr (std::is_same_v<Tdata, fp16_t> || std::is_same_v<Tdata, bf16_t>) {
                    sum += utils::cast<float>(in_c[j]);
                } else {
                    sum += (float)in_c[j];
                }
            }

            float avg = (klen > 0) ? (sum / (float)klen) : 0.0f;

            if constexpr (std::is_same_v<Tdata, fp16_t> || std::is_same_v<Tdata, bf16_t>) {
                out_c[i] = utils::cast<Tdata>(avg);
            } else {
                out_c[i] = (Tdata)avg;
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
        cpu::calculate<fp16_t>(_info, output, input);
        return INFINI_STATUS_SUCCESS;

    case INFINI_DTYPE_BF16:
        cpu::calculate<bf16_t>(_info, output, input);
        return INFINI_STATUS_SUCCESS;

    case INFINI_DTYPE_F32:
        cpu::calculate<float>(_info, output, input);
        return INFINI_STATUS_SUCCESS;

    case INFINI_DTYPE_F64:
        cpu::calculate<double>(_info, output, input);
        return INFINI_STATUS_SUCCESS;

    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::adaptive_avg_pool1d::cpu
