#include "pad_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../tensor.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace op::pad::nvidia {

static PadMode parseMode(const char *mode_str) {
    if (mode_str == nullptr) {
        return PadMode::CONSTANT;
    }
    if (std::strcmp(mode_str, "constant") == 0) {
        return PadMode::CONSTANT;
    }
    if (std::strcmp(mode_str, "reflect") == 0) {
        return PadMode::REFLECT;
    }
    if (std::strcmp(mode_str, "replicate") == 0) {
        return PadMode::REPLICATE;
    }
    if (std::strcmp(mode_str, "circular") == 0) {
        return PadMode::CIRCULAR;
    }
    return PadMode::CONSTANT;
}

static infiniStatus_t parsePadsTorchOrder(
    size_t ndim,
    const void *pad,
    size_t pad_size,
    std::vector<int> *pads_out) {
    if (pads_out == nullptr) {
        return INFINI_STATUS_BAD_PARAM;
    }
    if ((pad_size % sizeof(int)) != 0) {
        return INFINI_STATUS_BAD_PARAM;
    }
    if (pad_size != 0 && pad == nullptr) {
        return INFINI_STATUS_BAD_PARAM;
    }
    const int *pad_array = reinterpret_cast<const int *>(pad);
    const size_t pad_len = pad_size / sizeof(int);
    if (pad_len == 0 || (pad_len % 2) != 0 || pad_len > 2 * ndim) {
        return INFINI_STATUS_BAD_PARAM;
    }

    std::vector<int> pads(2 * ndim, 0);
    const size_t dims_padded = pad_len / 2;
    for (size_t j = 0; j < dims_padded; ++j) {
        const size_t dim = ndim - 1 - j;
        pads[2 * dim] = pad_array[2 * j];
        pads[2 * dim + 1] = pad_array[2 * j + 1];
    }
    *pads_out = std::move(pads);
    return INFINI_STATUS_SUCCESS;
}

Descriptor::~Descriptor() = default;

size_t Descriptor::workspaceSize() const {
    // Store metadata in device memory:
    // - input shape (ndim) + output shape (ndim)
    // - input strides (ndim) + output strides (ndim)
    // - pads (2 * ndim)
    // Use int64_t for simplicity/alignment.
    return sizeof(int64_t) * (6 * _ndim);
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    const void *pad,
    size_t pad_size,
    const char *mode,
    double value) {
    auto dtype = x_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_BF16);

    const size_t ndim = x_desc->ndim();
    if (ndim == 0) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }
    if (ndim > 16) {
        return INFINI_STATUS_BAD_PARAM;
    }

    std::vector<int> pads;
    auto st = parsePadsTorchOrder(ndim, pad, pad_size, &pads);
    if (st != INFINI_STATUS_SUCCESS) {
        return st;
    }

    auto x_shape = x_desc->shape();
    auto y_shape = y_desc->shape();

    std::vector<size_t> expected_output_shape = x_shape;
    for (size_t d = 0; d < ndim; ++d) {
        const int pad_left = pads[2 * d];
        const int pad_right = pads[2 * d + 1];
        if (pad_left < 0 || pad_right < 0) {
            return INFINI_STATUS_BAD_PARAM;
        }
        expected_output_shape[d] += static_cast<size_t>(pad_left + pad_right);
    }

    if (y_shape != expected_output_shape) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    const PadMode pad_mode = parseMode(mode);
    if (pad_mode == PadMode::REFLECT) {
        for (size_t d = 0; d < ndim; ++d) {
            const size_t in_size = x_shape[d];
            const int pad_left = pads[2 * d];
            const int pad_right = pads[2 * d + 1];
            if (in_size <= 1) {
                return INFINI_STATUS_BAD_PARAM;
            }
            if (pad_left >= static_cast<int>(in_size) || pad_right >= static_cast<int>(in_size)) {
                return INFINI_STATUS_BAD_PARAM;
            }
        }
    }

    *desc_ptr = new Descriptor(
        dtype,
        ndim,
        pad_mode,
        value,
        x_shape,
        x_desc->strides(),
        y_shape,
        y_desc->strides(),
        pads,
        y_desc->numel(),
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

template <typename T>
__device__ __forceinline__ T cast_pad_value(double v);

template <>
__device__ __forceinline__ half cast_pad_value<half>(double v) {
    return __float2half(static_cast<float>(v));
}

template <>
__device__ __forceinline__ cuda_bfloat16 cast_pad_value<cuda_bfloat16>(double v) {
    return __float2bfloat16_rn(static_cast<float>(v));
}

template <>
__device__ __forceinline__ float cast_pad_value<float>(double v) {
    return static_cast<float>(v);
}

template <>
__device__ __forceinline__ double cast_pad_value<double>(double v) {
    return v;
}

template <typename T>
__global__ void pad_kernel(
    T *y,
    const T *x,
    size_t ndim,
    const int64_t *in_shape,
    const int64_t *out_shape,
    const int64_t *in_strides,
    const int64_t *out_strides,
    const int64_t *pads,
    PadMode mode,
    double value,
    size_t out_numel) {
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= out_numel) {
        return;
    }

    // Compute logical output coordinates from flat index in row-major order.
    // This is independent of memory layout; memory offset uses out_strides.
    int64_t out_coords[16];
    int64_t in_coords[16];

    if (ndim > 16) {
        return;
    }

    size_t tmp = tid;
    for (size_t d = ndim; d-- > 0;) {
        const int64_t dim_size = out_shape[d];
        out_coords[d] = static_cast<int64_t>(tmp % static_cast<size_t>(dim_size));
        tmp /= static_cast<size_t>(dim_size);
    }

    bool inside = true;
    for (size_t d = 0; d < ndim; ++d) {
        const int64_t pad_left = pads[2 * d];
        const int64_t in_size = in_shape[d];
        const int64_t out_i = out_coords[d];
        int64_t in_i = out_i - pad_left;

        if (in_i < 0 || in_i >= in_size) {
            if (mode == PadMode::CONSTANT) {
                inside = false;
                break;
            }

            if (mode == PadMode::REPLICATE) {
                in_i = (in_i < 0) ? 0 : (in_size - 1);
            } else if (mode == PadMode::CIRCULAR) {
                // Wrap around
                int64_t m = in_i % in_size;
                if (m < 0) {
                    m += in_size;
                }
                in_i = m;
            } else if (mode == PadMode::REFLECT) {
                // Reflect around the edges, excluding the edge value.
                while (in_i < 0 || in_i >= in_size) {
                    if (in_i < 0) {
                        in_i = -in_i;
                    } else {
                        in_i = 2 * (in_size - 1) - in_i;
                    }
                }
            }
        }

        in_coords[d] = in_i;
    }

    int64_t out_off = 0;
    for (size_t d = 0; d < ndim; ++d) {
        out_off += out_coords[d] * out_strides[d];
    }

    if (!inside) {
        y[out_off] = cast_pad_value<T>(value);
        return;
    }

    int64_t in_off = 0;
    for (size_t d = 0; d < ndim; ++d) {
        in_off += in_coords[d] * in_strides[d];
    }

    y[out_off] = x[in_off];
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    void *stream) const {
    const size_t required = this->workspaceSize();
    if (workspace_size < required) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    // Pack metadata as int64_t arrays in device workspace.
    // Layout: in_shape[ndim], out_shape[ndim], in_strides[ndim], out_strides[ndim], pads[2*ndim]
    std::vector<int64_t> meta;
    meta.resize(6 * _ndim);

    int64_t *in_shape = meta.data();
    int64_t *out_shape = in_shape + _ndim;
    int64_t *in_strides = out_shape + _ndim;
    int64_t *out_strides = in_strides + _ndim;
    int64_t *pads = out_strides + _ndim;

    for (size_t d = 0; d < _ndim; ++d) {
        in_shape[d] = static_cast<int64_t>(_input_shape[d]);
        out_shape[d] = static_cast<int64_t>(_output_shape[d]);
        in_strides[d] = static_cast<int64_t>(_input_strides[d]);
        out_strides[d] = static_cast<int64_t>(_output_strides[d]);
    }
    for (size_t i = 0; i < 2 * _ndim; ++i) {
        pads[i] = static_cast<int64_t>(_pads[i]);
    }

    CHECK_CUDA(cudaMemcpyAsync(workspace, meta.data(), required, cudaMemcpyHostToDevice, cuda_stream));

    constexpr int BLOCK = 256;
    const int grid = static_cast<int>((_output_numel + BLOCK - 1) / BLOCK);

    const int64_t *d_in_shape = reinterpret_cast<const int64_t *>(workspace);
    const int64_t *d_out_shape = d_in_shape + _ndim;
    const int64_t *d_in_strides = d_out_shape + _ndim;
    const int64_t *d_out_strides = d_in_strides + _ndim;
    const int64_t *d_pads = d_out_strides + _ndim;

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        pad_kernel<half><<<grid, BLOCK, 0, cuda_stream>>>(
            reinterpret_cast<half *>(y),
            reinterpret_cast<const half *>(x),
            _ndim,
            d_in_shape,
            d_out_shape,
            d_in_strides,
            d_out_strides,
            d_pads,
            _mode,
            _value,
            _output_numel);
        break;
    case INFINI_DTYPE_BF16:
        pad_kernel<cuda_bfloat16><<<grid, BLOCK, 0, cuda_stream>>>(
            reinterpret_cast<cuda_bfloat16 *>(y),
            reinterpret_cast<const cuda_bfloat16 *>(x),
            _ndim,
            d_in_shape,
            d_out_shape,
            d_in_strides,
            d_out_strides,
            d_pads,
            _mode,
            _value,
            _output_numel);
        break;
    case INFINI_DTYPE_F32:
        pad_kernel<float><<<grid, BLOCK, 0, cuda_stream>>>(
            reinterpret_cast<float *>(y),
            reinterpret_cast<const float *>(x),
            _ndim,
            d_in_shape,
            d_out_shape,
            d_in_strides,
            d_out_strides,
            d_pads,
            _mode,
            _value,
            _output_numel);
        break;
    case INFINI_DTYPE_F64:
        pad_kernel<double><<<grid, BLOCK, 0, cuda_stream>>>(
            reinterpret_cast<double *>(y),
            reinterpret_cast<const double *>(x),
            _ndim,
            d_in_shape,
            d_out_shape,
            d_in_strides,
            d_out_strides,
            d_pads,
            _mode,
            _value,
            _output_numel);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::pad::nvidia
