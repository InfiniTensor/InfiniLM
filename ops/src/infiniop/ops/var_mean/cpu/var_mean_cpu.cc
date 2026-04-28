#include "var_mean_cpu.h"
#include "../../../../utils.h"
#include "../../../devices/cpu/common_cpu.h"
namespace op::var_mean::cpu {

Descriptor::~Descriptor() {}
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t var_output_desc,
    infiniopTensorDescriptor_t mean_output_desc,
    infiniopTensorDescriptor_t input_desc,
    size_t *dim,
    size_t dim_size,
    bool unbiased,
    bool keepdim) {
    auto result = VarMeanInfo::create(var_output_desc, input_desc, dim, dim_size, unbiased, keepdim);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(nullptr, result.take(), 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

// welford
namespace {
bool IsNanOut(const VarMeanInfo &info) {
    return (info.reduce_num == 0) || (info.reduce_num == 1 && info.unbiased_var == true);
}
// 直接用float计算
template <typename Tdata>
void computeVarMeanUsingWelfordCpu(const Tdata *input_ptr, float &var_output, float &mean_output, size_t start, size_t end, const VarMeanInfo &info) {
    if (start >= end) {
        return;
    }
    float old_mean = 0.0f; // previous mean
    float mean = 0.0f;     // new mean
    float M2 = 0.0f;       // variance sum
    size_t count = 0;      // element count of new sum
    for (size_t idx = start; idx < end; ++idx) {
        size_t input_offset = op::common_cpu::indexToOffset(idx, info.permuted_input_shape.size(), info.permuted_input_shape.data(), info.permuted_input_strides.data());
        ;
        float value = utils::cast<float>(input_ptr[input_offset]);
        count++;
        old_mean = mean;
        mean += (value - mean) / count;
        M2 += (value - old_mean) * (value - mean);
    }
    mean_output = mean;
    var_output = M2 / (info.unbiased_var ? (count - 1) : count);
}

template <typename Tdata>
infiniStatus_t calculateVarMean(
    const VarMeanInfo &info,
    Tdata *var_output,
    Tdata *mean_output,
    const Tdata *input) {
    Tdata nan_value = utils::cast<Tdata>(NAN);
    bool is_scalar = (info.reduce_dim_size == info.permuted_input_shape.size());
    // #pragma omp parallel for
    for (size_t i = 0; i < info.output_size; ++i) {
        size_t output_offset = op::common_cpu::indexToOffset(i, info.output_shape.size(), info.output_shape.data(), info.output_strides.data());
        if (IsNanOut(info)) {
            var_output[output_offset] = nan_value;
            if (info.reduce_num == 0) {
                mean_output[output_offset] = nan_value;
            } else {
                size_t input_idx = is_scalar ? 0 : i * info.reduce_num;
                size_t input_offset = op::common_cpu::indexToOffset(input_idx, info.permuted_input_shape.size(), info.permuted_input_shape.data(), info.permuted_input_strides.data());
                mean_output[output_offset] = input[input_offset];
            }
        } else {
            size_t start = is_scalar ? 0 : i * info.reduce_num;
            size_t end = is_scalar ? info.input_size : (i + 1) * info.reduce_num;
            float var = 0.0f, mean = 0.0f;
            computeVarMeanUsingWelfordCpu(input, var, mean, start, end, info);
            var_output[output_offset] = utils::cast<Tdata>(var);
            mean_output[output_offset] = utils::cast<Tdata>(mean);
        }
    }
    return INFINI_STATUS_SUCCESS;
}
} // namespace

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *var_output,
    void *mean_output,
    const void *input,
    bool unbiased,
    bool keepdim,
    void *stream) const {
    switch (_info.dtype) {
    case INFINI_DTYPE_F16:
        return calculateVarMean<fp16_t>(_info, (fp16_t *)var_output, (fp16_t *)mean_output, reinterpret_cast<const fp16_t *>(input));
    case INFINI_DTYPE_F32:
        return calculateVarMean<float>(_info, (float *)var_output, (float *)mean_output, reinterpret_cast<const float *>(input));
    case INFINI_DTYPE_BF16:
        return calculateVarMean<bf16_t>(_info, (bf16_t *)var_output, (bf16_t *)mean_output, reinterpret_cast<const bf16_t *>(input));
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::var_mean::cpu
