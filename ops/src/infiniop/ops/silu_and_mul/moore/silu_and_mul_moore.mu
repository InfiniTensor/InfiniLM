#include "../../../devices/moore/moore_common.h"
#include "../../../devices/moore/moore_handle.h"
#include "silu_and_mul_moore.h"

#include <musa_bf16.h>
#include <memory>

namespace op::silu_and_mul::moore {

struct Descriptor::Opaque {
    std::shared_ptr<device::moore::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc) {
    
    if (!desc_ptr) {
        return INFINI_STATUS_BAD_PARAM;
    }

    auto handle = reinterpret_cast<device::moore::Handle *>(handle_);
    auto dtype = y_desc->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);
    if (x_desc->dtype() != dtype) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    auto result = SiluAndMulInfo::create(y_desc, x_desc);
    CHECK_RESULT(result);
    auto info = result.take();

    *desc_ptr = new Descriptor(
        new Opaque{handle->internal()},
        std::move(info),
        0, 
        handle->device, handle->device_id);
        
    return INFINI_STATUS_SUCCESS;
}

template <typename T>
infiniStatus_t calculate_impl(
    const SiluAndMulInfo &info,
    std::shared_ptr<device::moore::Handle::Internal> &internal,
    void *y,
    const void *x,
    void *stream) {

    return internal->useMudnn(
        (musaStream_t)stream,
        [&](::musa::dnn::Handle &mudnn_handle) -> infiniStatus_t {
            
            ::musa::dnn::Tensor x_t, y_t;

            if constexpr (std::is_same_v<T, half>) {
                x_t.SetType(::musa::dnn::Tensor::Type::HALF);
                y_t.SetType(::musa::dnn::Tensor::Type::HALF);
            } else if constexpr (std::is_same_v<T, __mt_bfloat16>) {
                x_t.SetType(::musa::dnn::Tensor::Type::BFLOAT16);
                y_t.SetType(::musa::dnn::Tensor::Type::BFLOAT16);
            } else {
                x_t.SetType(::musa::dnn::Tensor::Type::FLOAT);
                y_t.SetType(::musa::dnn::Tensor::Type::FLOAT);
            }

            x_t.SetAddr(const_cast<void *>(x));
            y_t.SetAddr(y);

            // --- Construct 2D dimension information ---
            // Explicitly distinguish between Batch and Hidden dimensions
            int64_t b = static_cast<int64_t>(info.batch_size);
            int64_t h = static_cast<int64_t>(info.out_hidden_dim);

            // Input x logical shape is [batch, 2 * hidden]
            std::array<int64_t, 2> x_dims = {b, h * 2};
            std::array<int64_t, 2> x_strides = {h * 2, 1};

            // Output y logical shape is [batch, hidden]
            std::array<int64_t, 2> y_dims = {b, h};
            std::array<int64_t, 2> y_strides = {h, 1};

            x_t.SetNdInfo(2, x_dims.data(), x_strides.data());
            y_t.SetNdInfo(2, y_dims.data(), y_strides.data());

            // Invoke muDNN SwiGLU
            // muDNN will split each row (length 2*h) internally,
            // muDNN treats the first h elements of input x as the 'gate' 
            // and the following h elements as the 'up' projection.
            ::musa::dnn::SwiGlu swiglu;
            swiglu.Run(mudnn_handle, y_t, x_t);

            return INFINI_STATUS_SUCCESS;
        });
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *y, const void *x,
    void *stream) const {
    
    infiniDtype_t dtype = _info.dtype;

    switch (dtype) {
        case INFINI_DTYPE_F16:
            return calculate_impl<half>(_info, _opaque->internal, y, x, stream);
        case INFINI_DTYPE_F32:
            return calculate_impl<float>(_info, _opaque->internal, y, x, stream);
        case INFINI_DTYPE_BF16:
            return calculate_impl<__mt_bfloat16>(_info, _opaque->internal, y, x, stream);
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::silu_and_mul::moore
