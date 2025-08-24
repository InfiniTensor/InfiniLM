#include "linear_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

namespace op::linear::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t weight_desc,
    infiniopTensorDescriptor_t bias_desc) {
    
    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto dtype = output_desc->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    // Check that input and weight have the same dtype
    if (input_desc->dtype() != dtype || weight_desc->dtype() != dtype) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    // Check bias dtype if provided
    if (bias_desc && bias_desc->dtype() != dtype) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    auto result = LinearInfo::create(output_desc, input_desc, weight_desc, bias_desc);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        dtype, result.take(), 0,
        nullptr,
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata>
void calculate(
    const LinearInfo &info,
    void *output,
    const void *input,
    const void *weight,
    const void *bias) {

    auto output_ptr = reinterpret_cast<Tdata *>(output);
    auto input_ptr = reinterpret_cast<const Tdata *>(input);
    auto weight_ptr = reinterpret_cast<const Tdata *>(weight);
    auto bias_ptr = info.has_bias ? reinterpret_cast<const Tdata *>(bias) : nullptr;

#pragma omp parallel for
    for (ptrdiff_t batch_idx = 0; batch_idx < ptrdiff_t(info.batch_size); ++batch_idx) {
        for (size_t out_idx = 0; out_idx < info.out_features; ++out_idx) {
            // Calculate output[batch_idx, out_idx] = sum(input[batch_idx, :] * weight[out_idx, :]) + bias[out_idx]
            
            auto output_offset = batch_idx * info.output_batch_stride + out_idx * info.output_feature_stride;
            auto output_elem = output_ptr + output_offset;
            
            float sum = 0.0f;
            
            // Compute dot product: input[batch_idx, :] * weight[out_idx, :]
            for (size_t in_idx = 0; in_idx < info.in_features; ++in_idx) {
                auto input_offset = batch_idx * info.input_batch_stride + in_idx * info.input_feature_stride;
                auto weight_offset = out_idx * info.weight_out_stride + in_idx * info.weight_in_stride;
                
                auto input_elem = input_ptr + input_offset;
                auto weight_elem = weight_ptr + weight_offset;
                
                if constexpr (std::is_same<Tdata, fp16_t>::value || std::is_same<Tdata, bf16_t>::value) {
                    sum += utils::cast<float>(*input_elem) * utils::cast<float>(*weight_elem);
                } else {
                    sum += (*input_elem) * (*weight_elem);
                }
            }
            
            // Add bias if provided
            if (info.has_bias) {
                auto bias_elem = bias_ptr + out_idx * info.bias_stride;
                if constexpr (std::is_same<Tdata, fp16_t>::value || std::is_same<Tdata, bf16_t>::value) {
                    sum += utils::cast<float>(*bias_elem);
                } else {
                    sum += *bias_elem;
                }
            }
            
            // Store result
            if constexpr (std::is_same<Tdata, fp16_t>::value || std::is_same<Tdata, bf16_t>::value) {
                *output_elem = utils::cast<Tdata>(sum);
            } else {
                *output_elem = sum;
            }
        }
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    const void *weight,
    const void *bias,
    void *stream) const {

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        cpu::calculate<fp16_t>(_info, output, input, weight, bias);
        return INFINI_STATUS_SUCCESS;

    case INFINI_DTYPE_BF16:
        cpu::calculate<bf16_t>(_info, output, input, weight, bias);
        return INFINI_STATUS_SUCCESS;

    case INFINI_DTYPE_F32:
        cpu::calculate<float>(_info, output, input, weight, bias);
        return INFINI_STATUS_SUCCESS;

    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::linear::cpu