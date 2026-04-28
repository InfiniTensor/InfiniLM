#include "../../../../utils.h"
#include "avg_pool3d_nvidia.cuh"
#include <cudnn.h>
#include <limits>

namespace op::avg_pool3d::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
    cudnnTensorDescriptor_t x_desc = nullptr;
    cudnnTensorDescriptor_t y_desc = nullptr;
    cudnnPoolingDescriptor_t pool_desc = nullptr;
    size_t workspace_size = 0;

    Opaque(std::shared_ptr<device::nvidia::Handle::Internal> internal_ptr)
        : internal(internal_ptr) {}

    ~Opaque() {
        if (x_desc) {
            cudnnDestroyTensorDescriptor(x_desc);
        }
        if (y_desc) {
            cudnnDestroyTensorDescriptor(y_desc);
        }
        if (pool_desc) {
            cudnnDestroyPoolingDescriptor(pool_desc);
        }
    }
};

Descriptor::Descriptor(infiniDtype_t dtype, std::unique_ptr<Opaque> opaque,
                       infiniDevice_t device_type, int device_id)
    : InfiniopDescriptor{device_type, device_id},
      _opaque(std::move(opaque)),
      _dtype(dtype) {}

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    void *kernel_size,
    void *stride,
    void *padding) {

    auto nvidia_handle = reinterpret_cast<device::nvidia::Handle *>(handle);
    auto dtype = x_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_BF16);

    if (y_desc->dtype() != dtype) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    auto x_shape = x_desc->shape();
    auto y_shape = y_desc->shape();

    if (x_shape.size() != 5 || y_shape.size() != 5) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    if (!kernel_size) {
        return INFINI_STATUS_BAD_PARAM;
    }
    size_t *ks = reinterpret_cast<size_t *>(kernel_size);
    if (ks[0] == 0 || ks[1] == 0 || ks[2] == 0) {
        return INFINI_STATUS_BAD_PARAM;
    }
    size_t kernel_d = ks[0], kernel_h = ks[1], kernel_w = ks[2];

    size_t stride_d, stride_h, stride_w;
    if (stride) {
        size_t *s = reinterpret_cast<size_t *>(stride);
        stride_d = s[0];
        stride_h = s[1];
        stride_w = s[2];
    } else {
        stride_d = kernel_d;
        stride_h = kernel_h;
        stride_w = kernel_w;
    }

    size_t pad_d, pad_h, pad_w;
    if (padding) {
        size_t *p = reinterpret_cast<size_t *>(padding);
        pad_d = p[0];
        pad_h = p[1];
        pad_w = p[2];
    } else {
        pad_d = pad_h = pad_w = 0;
    }

    auto opaque = std::make_unique<Opaque>(nvidia_handle->internal());

    // Create cuDNN descriptors
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&opaque->x_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&opaque->y_desc));
    CHECK_CUDNN(cudnnCreatePoolingDescriptor(&opaque->pool_desc));

    // Set tensor descriptors
    int n = static_cast<int>(x_shape[0]);
    int c = static_cast<int>(x_shape[1]);
    int d = static_cast<int>(x_shape[2]);
    int h = static_cast<int>(x_shape[3]);
    int w = static_cast<int>(x_shape[4]);
    int out_d = static_cast<int>(y_shape[2]);
    int out_h = static_cast<int>(y_shape[3]);
    int out_w = static_cast<int>(y_shape[4]);

    int input_dims[5] = {n, c, d, h, w};
    auto x_strides = x_desc->strides();
    if (x_strides.size() != 5) {
        return INFINI_STATUS_BAD_TENSOR_STRIDES;
    }
    int input_strides[5] = {};
    for (size_t i = 0; i < 5; ++i) {
        if (x_strides[i] <= 0 || x_strides[i] > std::numeric_limits<int>::max()) {
            return INFINI_STATUS_BAD_TENSOR_STRIDES;
        }
        input_strides[i] = static_cast<int>(x_strides[i]);
    }

    int output_dims[5] = {n, c, out_d, out_h, out_w};
    auto y_strides = y_desc->strides();
    if (y_strides.size() != 5) {
        return INFINI_STATUS_BAD_TENSOR_STRIDES;
    }
    int output_strides[5] = {};
    for (size_t i = 0; i < 5; ++i) {
        if (y_strides[i] <= 0 || y_strides[i] > std::numeric_limits<int>::max()) {
            return INFINI_STATUS_BAD_TENSOR_STRIDES;
        }
        output_strides[i] = static_cast<int>(y_strides[i]);
    }

    cudnnDataType_t cudnn_dtype = device::nvidia::getCudnnDtype(dtype);
    CHECK_CUDNN(cudnnSetTensorNdDescriptor(
        opaque->x_desc, cudnn_dtype, 5, input_dims, input_strides));
    CHECK_CUDNN(cudnnSetTensorNdDescriptor(
        opaque->y_desc, cudnn_dtype, 5, output_dims, output_strides));

    // Set pooling descriptor
    int window_dims[3] = {static_cast<int>(kernel_d), static_cast<int>(kernel_h), static_cast<int>(kernel_w)};
    int padding_dims[3] = {static_cast<int>(pad_d), static_cast<int>(pad_h), static_cast<int>(pad_w)};
    int stride_dims[3] = {static_cast<int>(stride_d), static_cast<int>(stride_h), static_cast<int>(stride_w)};

    CHECK_CUDNN(cudnnSetPoolingNdDescriptor(
        opaque->pool_desc,
        CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
        CUDNN_NOT_PROPAGATE_NAN,
        3, window_dims, padding_dims, stride_dims));

    *desc_ptr = new Descriptor(dtype, std::move(opaque), handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

size_t Descriptor::workspaceSize() const {
    return _opaque->workspace_size;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    void *stream) const {

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    return _opaque->internal->useCudnn(cuda_stream, [&](cudnnHandle_t cudnn_handle) {
        const void *alpha = nullptr;
        const void *beta = nullptr;
        if (_dtype == INFINI_DTYPE_F32) {
            static const float alpha_val = 1.0f, beta_val = 0.0f;
            alpha = &alpha_val;
            beta = &beta_val;
        } else if (_dtype == INFINI_DTYPE_F64) {
            static const double alpha_val = 1.0, beta_val = 0.0;
            alpha = &alpha_val;
            beta = &beta_val;
        } else {
            // For F16/BF16, use float alpha/beta
            static const float alpha_val = 1.0f, beta_val = 0.0f;
            alpha = &alpha_val;
            beta = &beta_val;
        }

        CHECK_CUDNN(cudnnPoolingForward(
            cudnn_handle,
            _opaque->pool_desc,
            alpha,
            _opaque->x_desc, x,
            beta,
            _opaque->y_desc, y));

        return INFINI_STATUS_SUCCESS;
    });
}

} // namespace op::avg_pool3d::nvidia
