#include "linear_backwards_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

namespace op::linear_backwards::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t grad_input_desc,
    infiniopTensorDescriptor_t grad_weight_desc,
    infiniopTensorDescriptor_t grad_bias_desc,
    infiniopTensorDescriptor_t grad_output_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t weight_desc) {
    
    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto dtype = grad_output_desc->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    // Check that all tensors have the same dtype
    if (input_desc->dtype() != dtype || weight_desc->dtype() != dtype) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    if (grad_input_desc && grad_input_desc->dtype() != dtype) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    if (grad_weight_desc && grad_weight_desc->dtype() != dtype) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    if (grad_bias_desc && grad_bias_desc->dtype() != dtype) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    auto result = LinearBackwardsInfo::create(grad_input_desc, grad_weight_desc, grad_bias_desc,
                                              grad_output_desc, input_desc, weight_desc);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        dtype, result.take(), 0,
        nullptr,
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata>
void calculate(
    const LinearBackwardsInfo &info,
    void *grad_input,
    void *grad_weight,
    void *grad_bias,
    const void *grad_output,
    const void *input,
    const void *weight) {

    auto grad_output_ptr = reinterpret_cast<const Tdata *>(grad_output);
    auto input_ptr = reinterpret_cast<const Tdata *>(input);
    auto weight_ptr = reinterpret_cast<const Tdata *>(weight);

    auto grad_input_ptr = grad_input ? reinterpret_cast<Tdata *>(grad_input) : nullptr;
    auto grad_weight_ptr = grad_weight ? reinterpret_cast<Tdata *>(grad_weight) : nullptr;
    auto grad_bias_ptr = grad_bias ? reinterpret_cast<Tdata *>(grad_bias) : nullptr;

    // Compute grad_input = grad_output @ weight
    if (grad_input_ptr) {
#pragma omp parallel for
        for (ptrdiff_t batch_idx = 0; batch_idx < ptrdiff_t(info.batch_size); ++batch_idx) {
            for (size_t in_idx = 0; in_idx < info.in_features; ++in_idx) {
                auto grad_input_offset = batch_idx * info.grad_input_batch_stride + in_idx * info.grad_input_feature_stride;
                auto grad_input_elem = grad_input_ptr + grad_input_offset;
                
                float sum = 0.0f;
                
                // grad_input[batch_idx, in_idx] = sum(grad_output[batch_idx, :] * weight[:, in_idx])
                for (size_t out_idx = 0; out_idx < info.out_features; ++out_idx) {
                    auto grad_output_offset = batch_idx * info.grad_output_batch_stride + out_idx * info.grad_output_feature_stride;
                    auto weight_offset = out_idx * info.weight_out_stride + in_idx * info.weight_in_stride;
                    
                    auto grad_output_elem = grad_output_ptr + grad_output_offset;
                    auto weight_elem = weight_ptr + weight_offset;
                    
                    if constexpr (std::is_same<Tdata, fp16_t>::value || std::is_same<Tdata, bf16_t>::value) {
                        sum += utils::cast<float>(*grad_output_elem) * utils::cast<float>(*weight_elem);
                    } else {
                        sum += (*grad_output_elem) * (*weight_elem);
                    }
                }
                
                if constexpr (std::is_same<Tdata, fp16_t>::value || std::is_same<Tdata, bf16_t>::value) {
                    *grad_input_elem = utils::cast<Tdata>(sum);
                } else {
                    *grad_input_elem = sum;
                }
            }
        }
    }

    // Compute grad_weight = grad_output.T @ input
    if (grad_weight_ptr) {
        // Initialize grad_weight to zero
#pragma omp parallel for
        for (size_t out_idx = 0; out_idx < info.out_features; ++out_idx) {
            for (size_t in_idx = 0; in_idx < info.in_features; ++in_idx) {
                auto grad_weight_offset = out_idx * info.grad_weight_out_stride + in_idx * info.grad_weight_in_stride;
                auto grad_weight_elem = grad_weight_ptr + grad_weight_offset;
                *grad_weight_elem = utils::cast<Tdata>(0.0f);
            }
        }

        // Accumulate grad_weight
#pragma omp parallel for
        for (size_t out_idx = 0; out_idx < info.out_features; ++out_idx) {
            for (size_t in_idx = 0; in_idx < info.in_features; ++in_idx) {
                auto grad_weight_offset = out_idx * info.grad_weight_out_stride + in_idx * info.grad_weight_in_stride;
                auto grad_weight_elem = grad_weight_ptr + grad_weight_offset;
                
                float sum = 0.0f;
                
                // grad_weight[out_idx, in_idx] = sum(grad_output[:, out_idx] * input[:, in_idx])
                for (size_t batch_idx = 0; batch_idx < info.batch_size; ++batch_idx) {
                    auto grad_output_offset = batch_idx * info.grad_output_batch_stride + out_idx * info.grad_output_feature_stride;
                    auto input_offset = batch_idx * info.input_batch_stride + in_idx * info.input_feature_stride;
                    
                    auto grad_output_elem = grad_output_ptr + grad_output_offset;
                    auto input_elem = input_ptr + input_offset;
                    
                    if constexpr (std::is_same<Tdata, fp16_t>::value || std::is_same<Tdata, bf16_t>::value) {
                        sum += utils::cast<float>(*grad_output_elem) * utils::cast<float>(*input_elem);
                    } else {
                        sum += (*grad_output_elem) * (*input_elem);
                    }
                }
                
                if constexpr (std::is_same<Tdata, fp16_t>::value || std::is_same<Tdata, bf16_t>::value) {
                    *grad_weight_elem = utils::cast<Tdata>(sum);
                } else {
                    *grad_weight_elem = sum;
                }
            }
        }
    }

    // Compute grad_bias = sum(grad_output, dim=0)
    if (grad_bias_ptr && info.has_bias) {
#pragma omp parallel for
        for (size_t out_idx = 0; out_idx < info.out_features; ++out_idx) {
            auto grad_bias_offset = out_idx * info.grad_bias_stride;
            auto grad_bias_elem = grad_bias_ptr + grad_bias_offset;
            
            float sum = 0.0f;
            
            // grad_bias[out_idx] = sum(grad_output[:, out_idx])
            for (size_t batch_idx = 0; batch_idx < info.batch_size; ++batch_idx) {
                auto grad_output_offset = batch_idx * info.grad_output_batch_stride + out_idx * info.grad_output_feature_stride;
                auto grad_output_elem = grad_output_ptr + grad_output_offset;
                
                if constexpr (std::is_same<Tdata, fp16_t>::value || std::is_same<Tdata, bf16_t>::value) {
                    sum += utils::cast<float>(*grad_output_elem);
                } else {
                    sum += *grad_output_elem;
                }
            }
            
            if constexpr (std::is_same<Tdata, fp16_t>::value || std::is_same<Tdata, bf16_t>::value) {
                *grad_bias_elem = utils::cast<Tdata>(sum);
            } else {
                *grad_bias_elem = sum;
            }
        }
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *grad_input,
    void *grad_weight,
    void *grad_bias,
    const void *grad_output,
    const void *input,
    const void *weight,
    void *stream) const {

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        cpu::calculate<fp16_t>(_info, grad_input, grad_weight, grad_bias, grad_output, input, weight);
        return INFINI_STATUS_SUCCESS;

    case INFINI_DTYPE_BF16:
        cpu::calculate<bf16_t>(_info, grad_input, grad_weight, grad_bias, grad_output, input, weight);
        return INFINI_STATUS_SUCCESS;

    case INFINI_DTYPE_F32:
        cpu::calculate<float>(_info, grad_input, grad_weight, grad_bias, grad_output, input, weight);
        return INFINI_STATUS_SUCCESS;

    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::linear_backwards::cpu