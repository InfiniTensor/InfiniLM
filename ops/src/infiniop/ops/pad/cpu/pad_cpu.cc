#include "pad_cpu.h"
#include "../../../tensor.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace op::pad::cpu {

PadMode parseMode(const char *mode_str) {
    if (mode_str == nullptr) {
        return PadMode::CONSTANT;
    }
    if (std::strcmp(mode_str, "constant") == 0) {
        return PadMode::CONSTANT;
    } else if (std::strcmp(mode_str, "reflect") == 0) {
        return PadMode::REFLECT;
    } else if (std::strcmp(mode_str, "replicate") == 0) {
        return PadMode::REPLICATE;
    } else if (std::strcmp(mode_str, "circular") == 0) {
        return PadMode::CIRCULAR;
    }
    return PadMode::CONSTANT; // Default
}

utils::Result<PadInfo> PadInfo::create(
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t y_desc,
    const void *pad,
    size_t pad_size,
    const char *mode_str,
    double value) {

    auto x_shape = x_desc->shape();
    auto y_shape = y_desc->shape();
    size_t ndim = x_desc->ndim();

    // Parse pad array
    if ((pad_size % sizeof(int)) != 0) {
        return INFINI_STATUS_BAD_PARAM;
    }
    if (pad_size != 0 && pad == nullptr) {
        return INFINI_STATUS_BAD_PARAM;
    }
    const int *pad_array = reinterpret_cast<const int *>(pad);
    size_t pad_len = pad_size / sizeof(int);

    // Padding follows PyTorch order:
    // (pad_left_last_dim, pad_right_last_dim, pad_left_second_last, pad_right_second_last, ...)
    // and applies to the last dimensions first.
    std::vector<int> pads(2 * ndim, 0);
    if (pad_len == 0 || (pad_len % 2) != 0 || pad_len > 2 * ndim) {
        return INFINI_STATUS_BAD_PARAM;
    }
    size_t dims_padded = pad_len / 2;
    for (size_t j = 0; j < dims_padded; ++j) {
        size_t dim = ndim - 1 - j;
        pads[2 * dim] = pad_array[2 * j];
        pads[2 * dim + 1] = pad_array[2 * j + 1];
    }

    for (size_t i = 0; i < ndim; ++i) {
        if (pads[2 * i] < 0 || pads[2 * i + 1] < 0) {
            return INFINI_STATUS_BAD_PARAM;
        }
    }

    // Calculate expected output shape
    std::vector<size_t> expected_output_shape = x_shape;
    for (size_t i = 0; i < ndim; ++i) {
        expected_output_shape[i] += pads[2 * i] + pads[2 * i + 1];
    }

    if (y_shape != expected_output_shape) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    const PadMode mode = parseMode(mode_str);
    if (mode == PadMode::REFLECT) {
        for (size_t i = 0; i < ndim; ++i) {
            const int64_t in_size = static_cast<int64_t>(x_shape[i]);
            const int64_t pad_left = static_cast<int64_t>(pads[2 * i]);
            const int64_t pad_right = static_cast<int64_t>(pads[2 * i + 1]);
            if (pad_left == 0 && pad_right == 0) {
                continue;
            }
            if (in_size <= 1) {
                return INFINI_STATUS_BAD_PARAM;
            }
            if (pad_left >= in_size || pad_right >= in_size) {
                return INFINI_STATUS_BAD_PARAM;
            }
        }
    }

    PadInfo info;
    info.ndim = ndim;
    info.input_shape = x_shape;
    info.input_strides = x_desc->strides();
    info.output_shape = y_shape;
    info.output_strides = y_desc->strides();
    info.pads = pads;
    info.mode = mode;
    info.value = value;

    return utils::Result<PadInfo>(std::move(info));
}

Descriptor::~Descriptor() = default;

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

    auto info_result = PadInfo::create(x_desc, y_desc, pad, pad_size, mode, value);
    CHECK_RESULT(info_result);

    *desc_ptr = new Descriptor(dtype, info_result.take(), handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename T>
void pad_impl(
    const PadInfo &info,
    T *y,
    const T *x) {

    size_t out_numel = 1;
    for (size_t i = 0; i < info.ndim; ++i) {
        out_numel *= info.output_shape[i];
    }

    const T pad_value = utils::cast<T>(info.value);

    std::vector<int64_t> out_coords(info.ndim);
    std::vector<int64_t> in_coords(info.ndim);

    for (size_t linear = 0; linear < out_numel; ++linear) {
        // Convert linear index to logical coordinates in row-major order.
        size_t tmp = linear;
        for (size_t d = info.ndim; d-- > 0;) {
            out_coords[d] = static_cast<int64_t>(tmp % info.output_shape[d]);
            tmp /= info.output_shape[d];
        }

        bool inside = true;
        for (size_t d = 0; d < info.ndim; ++d) {
            const int64_t pad_left = static_cast<int64_t>(info.pads[2 * d]);
            const int64_t in_size = static_cast<int64_t>(info.input_shape[d]);
            const int64_t out_i = out_coords[d];
            int64_t in_i = out_i - pad_left;

            if (in_i < 0 || in_i >= in_size) {
                if (info.mode == PadMode::CONSTANT) {
                    inside = false;
                    break;
                }

                if (info.mode == PadMode::REPLICATE) {
                    in_i = (in_i < 0) ? 0 : (in_size - 1);
                } else if (info.mode == PadMode::CIRCULAR) {
                    int64_t m = in_i % in_size;
                    if (m < 0) {
                        m += in_size;
                    }
                    in_i = m;
                } else if (info.mode == PadMode::REFLECT) {
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

        ptrdiff_t out_off = 0;
        for (size_t d = 0; d < info.ndim; ++d) {
            out_off += static_cast<ptrdiff_t>(out_coords[d]) * info.output_strides[d];
        }

        if (!inside) {
            *(y + out_off) = pad_value;
            continue;
        }

        ptrdiff_t in_off = 0;
        for (size_t d = 0; d < info.ndim; ++d) {
            in_off += static_cast<ptrdiff_t>(in_coords[d]) * info.input_strides[d];
        }

        *(y + out_off) = *(x + in_off);
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    void *stream) const {

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        pad_impl<fp16_t>(_info, reinterpret_cast<fp16_t *>(y), reinterpret_cast<const fp16_t *>(x));
        break;
    case INFINI_DTYPE_BF16:
        pad_impl<bf16_t>(_info, reinterpret_cast<bf16_t *>(y), reinterpret_cast<const bf16_t *>(x));
        break;
    case INFINI_DTYPE_F32:
        pad_impl<float>(_info, reinterpret_cast<float *>(y), reinterpret_cast<const float *>(x));
        break;
    case INFINI_DTYPE_F64:
        pad_impl<double>(_info, reinterpret_cast<double *>(y), reinterpret_cast<const double *>(x));
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::pad::cpu
