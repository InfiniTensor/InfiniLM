#include "embedding_cpu.h"
#include "../../../../utils.h"
#include "../../../handle.h"
#include "../../../tensor.h"
#include <cstring>

namespace op::embedding::cpu {

struct Descriptor::Opaque {};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t weight_desc) {

    auto input_shape = input_desc->shape();
    auto weight_shape = weight_desc->shape();

    CHECK_OR_RETURN(weight_shape.size() == 2, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(output_desc->shape().size() == input_shape.size() + 1, INFINI_STATUS_BAD_TENSOR_SHAPE);

    auto output_shape = output_desc->shape();
    size_t embedding_dim = weight_shape[1];
    CHECK_OR_RETURN(output_shape.back() == embedding_dim, INFINI_STATUS_BAD_TENSOR_SHAPE);

    for (size_t i = 0; i < input_shape.size(); ++i) {
        CHECK_OR_RETURN(output_shape[i] == input_shape[i], INFINI_STATUS_BAD_TENSOR_SHAPE);
    }

    auto input_dtype = input_desc->dtype();
    auto weight_dtype = weight_desc->dtype();
    CHECK_OR_RETURN(input_dtype == INFINI_DTYPE_I32 || input_dtype == INFINI_DTYPE_I64,
                    INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(weight_dtype == INFINI_DTYPE_F32 || weight_dtype == INFINI_DTYPE_F16 || weight_dtype == INFINI_DTYPE_BF16, INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(output_desc->dtype() == weight_dtype, INFINI_STATUS_BAD_TENSOR_DTYPE);

    size_t num_indices = 1;
    for (auto dim : input_shape) {
        num_indices *= dim;
    }

    size_t vocab_size = weight_shape[0];

    *desc_ptr = new Descriptor(
        num_indices,
        embedding_dim,
        vocab_size,
        input_dtype,
        weight_dtype,
        new Opaque{},
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *output,
    const void *input,
    const void *weight,
    void *stream) const {

    if (_num_indices == 0) {
        return INFINI_STATUS_SUCCESS;
    }

    size_t element_size = infiniSizeOf(_weight_dtype);
    size_t row_bytes = _embedding_dim * element_size;

    if (_input_dtype == INFINI_DTYPE_I32) {
        const int32_t *indices_ptr = reinterpret_cast<const int32_t *>(input);
        const std::byte *weight_ptr = reinterpret_cast<const std::byte *>(weight);
        std::byte *out_ptr = reinterpret_cast<std::byte *>(output);

        for (size_t i = 0; i < _num_indices; ++i) {
            int32_t idx = indices_ptr[i];
            if (idx >= 0 && static_cast<size_t>(idx) < _vocab_size) {
                std::memcpy(out_ptr + i * row_bytes,
                            weight_ptr + static_cast<size_t>(idx) * row_bytes,
                            row_bytes);
            }
        }
    } else if (_input_dtype == INFINI_DTYPE_I64) {
        const int64_t *indices_ptr = reinterpret_cast<const int64_t *>(input);
        const std::byte *weight_ptr = reinterpret_cast<const std::byte *>(weight);
        std::byte *out_ptr = reinterpret_cast<std::byte *>(output);

        for (size_t i = 0; i < _num_indices; ++i) {
            int64_t idx = indices_ptr[i];
            if (idx >= 0 && static_cast<size_t>(idx) < _vocab_size) {
                std::memcpy(out_ptr + i * row_bytes,
                            weight_ptr + static_cast<size_t>(idx) * row_bytes,
                            row_bytes);
            }
        }
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::embedding::cpu
