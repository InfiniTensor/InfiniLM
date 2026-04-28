#include "layer_norm_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../reduce/cpu/reduce.h"
#include "../info.h"

namespace op::layer_norm::cpu {

template <typename Tdata>
infiniStatus_t calculate_layer_norm(
    const LayerNormInfo &info,
    Tdata *output,
    Tdata *input_standardization,
    Tdata *input_std_deviation,
    const Tdata *input,
    const Tdata *weight,
    const Tdata *bias) {

    const size_t ndim = info.ndim;
    const size_t norm_size = info.normalized_size; // last dim
    const size_t othersize = info.othersize;

    const auto &shape = info.input_shape;
    const auto &in_strides = info.input_strides;
    const auto &out_strides = info.output_strides;
    const auto &std_strides = info.input_standardization_strides;
    const auto &stddev_strides = info.input_std_deviation_strides;

    // -------- Special fast path: 1D tensor --------
    if (ndim == 1) {
        const Tdata *input_ptr = input;
        Tdata *output_ptr = output;
        Tdata *standard_ptr = input_standardization;
        Tdata *std_ptr = input_std_deviation;

        float mean = op::common_cpu::reduce_op::sum(
                         input_ptr,
                         norm_size,
                         in_strides[0])
                   / norm_size;

        float sum_sq = op::common_cpu::reduce_op::sumSquared(
            input_ptr,
            norm_size,
            in_strides[0]);

        float var = sum_sq / norm_size - mean * mean;
        float std_dev = std::sqrt(var + info.eps);

        *std_ptr = utils::cast<Tdata>(std_dev);

        for (size_t d = 0; d < norm_size; d++) {
            float x = utils::cast<float>(
                *(input_ptr + d * in_strides[0]));

            float x_std = (x - mean) / std_dev;

            *(standard_ptr + d * std_strides[0]) = utils::cast<Tdata>(x_std);

            float w = utils::cast<float>(
                *(weight + d * info.weight_strides[0]));

            float bval = info.bias_exist
                           ? utils::cast<float>(
                               *(bias + d * info.bias_strides[0]))
                           : 0.0f;

            *(output_ptr + d * out_strides[0]) = utils::cast<Tdata>(x_std * w + bval);
        }

        return INFINI_STATUS_SUCCESS;
    }

    // -------- General case: ndim >= 2 --------

    // index for dims [0 ... ndim-2]
    std::vector<size_t> idx(ndim - 1, 0);

#pragma omp parallel for
    for (ptrdiff_t b = 0; b < (ptrdiff_t)othersize; b++) {

        // ---- compute base offsets ----
        ptrdiff_t in_offset = 0;
        ptrdiff_t out_offset = 0;
        ptrdiff_t std_offset = 0;
        ptrdiff_t stddev_offset = 0;

        for (size_t d = 0; d < ndim - 1; d++) {
            in_offset += idx[d] * in_strides[d];
            out_offset += idx[d] * out_strides[d];
            std_offset += idx[d] * std_strides[d];
            stddev_offset += idx[d] * stddev_strides[d];
        }

        const Tdata *input_ptr = input + in_offset;
        Tdata *output_ptr = output + out_offset;
        Tdata *standard_ptr = input_standardization + std_offset;
        Tdata *std_ptr = input_std_deviation + stddev_offset;

        // ---- mean ----
        float mean = op::common_cpu::reduce_op::sum(
                         input_ptr,
                         norm_size,
                         in_strides[ndim - 1])
                   / norm_size;

        // ---- variance ----
        float sum_sq = op::common_cpu::reduce_op::sumSquared(
            input_ptr,
            norm_size,
            in_strides[ndim - 1]);

        float var = sum_sq / norm_size - mean * mean;
        float std_dev = std::sqrt(var + info.eps);

        *std_ptr = utils::cast<Tdata>(std_dev);

        // ---- normalize ----
        for (size_t d = 0; d < norm_size; d++) {
            float x = utils::cast<float>(
                *(input_ptr + d * in_strides[ndim - 1]));

            float x_std = (x - mean) / std_dev;

            *(standard_ptr + d * std_strides[ndim - 1]) = utils::cast<Tdata>(x_std);

            float w = utils::cast<float>(
                *(weight + d * info.weight_strides[0]));

            float bval = info.bias_exist
                           ? utils::cast<float>(
                               *(bias + d * info.bias_strides[0]))
                           : 0.0f;

            *(output_ptr + d * out_strides[ndim - 1]) = utils::cast<Tdata>(x_std * w + bval);
        }

        // ---- increment multi-dim index (odometer style) ----
        for (int d = (int)ndim - 2; d >= 0; d--) {
            idx[d]++;
            if (idx[d] < shape[d]) {
                break;
            }
            idx[d] = 0;
        }
    }

    return INFINI_STATUS_SUCCESS;
}

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_standardization_desc,
    infiniopTensorDescriptor_t input_std_deviation_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t weight_desc,
    infiniopTensorDescriptor_t bias_desc,
    float eps) {
    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);

    //  --------------------- start: check data type and calculate workspace size ----------------------
    auto dtype = input_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);
    size_t WorkSpaceSize = 0;

    auto result = LayerNormInfo::createLayerNormInfo(
        output_desc,
        input_standardization_desc,
        input_std_deviation_desc,
        input_desc,
        weight_desc,
        bias_desc,
        eps);
    CHECK_RESULT(result);
    const LayerNormInfo &info = result.take();

    *desc_ptr = new Descriptor(
        dtype, std::move(info), WorkSpaceSize,
        nullptr,
        handle->device, handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

#define CALCULATE_LAYER_NORM(TDATA)                                          \
    CHECK_STATUS(calculate_layer_norm<TDATA>(_info,                          \
                                             (TDATA *)output,                \
                                             (TDATA *)input_standardization, \
                                             (TDATA *)input_std_deviation,   \
                                             (const TDATA *)input,           \
                                             (const TDATA *)weight,          \
                                             (const TDATA *)bias))

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    void *input_standardization,
    void *input_std_deviation,
    const void *input,
    const void *weight,
    const void *bias,
    void *stream) const {
    if (_info.dtype == INFINI_DTYPE_F16) {
        CALCULATE_LAYER_NORM(fp16_t);
    } else if (_info.dtype == INFINI_DTYPE_BF16) {
        CALCULATE_LAYER_NORM(bf16_t);
    } else if (_info.dtype == INFINI_DTYPE_F32) {
        CALCULATE_LAYER_NORM(float);
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    return INFINI_STATUS_SUCCESS;
}
} // namespace op::layer_norm::cpu
