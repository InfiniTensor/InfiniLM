#include "hardtanh_cpu.h"

#include <type_traits>

namespace op::hardtanh::cpu {

Descriptor::Descriptor(infiniDtype_t dtype,
                       op::elementwise::ElementwiseInfo info,
                       size_t workspace_size,
                       infiniDevice_t device_type,
                       int device_id,
                       float min_val,
                       float max_val)
    : InfiniopDescriptor{device_type, device_id},
      _dtype(dtype),
      _info(std::move(info)),
      _workspace_size(workspace_size),
      _min_val(min_val),
      _max_val(max_val) {}

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec,
    float min_val,
    float max_val) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto dtype = out_desc->dtype();

    const auto &input_desc = input_desc_vec.at(0);
    const auto &output_shape = out_desc->shape();
    const auto &input_shape = input_desc->shape();

    CHECK_DTYPE(dtype, INFINI_DTYPE_BF16, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);
    CHECK_SAME_SHAPE(output_shape, input_shape);

    auto info_result = op::elementwise::ElementwiseInfo::create(out_desc, input_desc_vec);
    CHECK_RESULT(info_result);

    *desc_ptr = new Descriptor(
        dtype,
        info_result.take(),
        0,
        handle->device,
        handle->device_id,
        min_val,
        max_val);

    return INFINI_STATUS_SUCCESS;
}

template <typename T>
static infiniStatus_t launchCpuHardTanh(const op::elementwise::ElementwiseInfo &info,
                                        void *output,
                                        const std::vector<const void *> &inputs,
                                        float min_val,
                                        float max_val) {
    if (inputs.empty()) {
        return INFINI_STATUS_BAD_PARAM;
    }

    T *out = reinterpret_cast<T *>(output);
    const T *in = reinterpret_cast<const T *>(inputs[0]);
    const auto ndim = info.getNdim();
    const auto *output_shape = info.getOutputShape();
    const auto *output_strides = info.getOutputStrides();
    const auto *input_shape = info.getInputShape(0);
    const auto *input_strides = info.getInputStrides(0);
    const auto *input_contiguous = info.getInputContiguous();
    ptrdiff_t output_size = info.getOutputSize();

#pragma omp parallel for if (output_size > 1024)
    for (ptrdiff_t i = 0; i < output_size; ++i) {
        const size_t out_idx = info.isOutputContiguous()
                                 ? static_cast<size_t>(i)
                                 : op::common_cpu::indexToOffset(i, ndim, output_shape, output_strides);
        const size_t in_idx = input_contiguous[0]
                                ? static_cast<size_t>(i)
                                : op::common_cpu::indexToOffset(i, ndim, input_shape, input_strides);

        if constexpr (std::is_same_v<T, fp16_t> || std::is_same_v<T, bf16_t>) {
            float value = utils::cast<float>(in[in_idx]);
            float clamped = HardTanhOp{}(value, min_val, max_val);
            out[out_idx] = utils::cast<T>(clamped);
        } else {
            out[out_idx] = HardTanhOp{}(in[in_idx], min_val, max_val);
        }
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const {
    (void)workspace;
    (void)workspace_size;
    (void)stream;

    if (inputs.size() != 1) {
        return INFINI_STATUS_BAD_PARAM;
    }

    switch (_dtype) {
    case INFINI_DTYPE_BF16:
        return launchCpuHardTanh<bf16_t>(_info, output, inputs, _min_val, _max_val);
    case INFINI_DTYPE_F16:
        return launchCpuHardTanh<fp16_t>(_info, output, inputs, _min_val, _max_val);
    case INFINI_DTYPE_F32:
        return launchCpuHardTanh<float>(_info, output, inputs, _min_val, _max_val);
    case INFINI_DTYPE_F64:
        return launchCpuHardTanh<double>(_info, output, inputs, _min_val, _max_val);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}
} // namespace op::hardtanh::cpu
