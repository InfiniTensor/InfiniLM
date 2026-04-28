#include "interpolate_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace op::interpolate::cpu {

inline size_t offset_3d(const std::vector<ptrdiff_t> &s,
                        size_t b, size_t c, size_t x) {
    return b * s[0] + c * s[1] + x * s[2];
}

inline size_t offset_4d(const std::vector<ptrdiff_t> &s,
                        size_t b, size_t c, size_t h, size_t w) {
    return b * s[0] + c * s[1] + h * s[2] + w * s[3];
}

static bool try_parse_mode(const char *mode_str, InterpolateMode &mode) {
    if (std::strcmp(mode_str, "nearest") == 0) {
        mode = InterpolateMode::NEAREST;
        return true;
    } else if (std::strcmp(mode_str, "linear") == 0) {
        mode = InterpolateMode::LINEAR;
        return true;
    } else if (std::strcmp(mode_str, "bilinear") == 0) {
        mode = InterpolateMode::BILINEAR;
        return true;
    } else if (std::strcmp(mode_str, "trilinear") == 0) {
        mode = InterpolateMode::TRILINEAR;
        return true;
    } else if (std::strcmp(mode_str, "area") == 0) {
        mode = InterpolateMode::AREA;
        return true;
    }
    return false;
}

static double compute_scale(size_t in_size, size_t out_size, int align_corners) {
    if (out_size == 0) {
        return 0.0;
    }
    if (align_corners) {
        return (out_size > 1) ? (static_cast<double>(in_size) - 1.0) / (static_cast<double>(out_size) - 1.0) : 0.0;
    }
    return static_cast<double>(in_size) / static_cast<double>(out_size);
}

utils::Result<InterpolateInfo> InterpolateInfo::create(
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t y_desc,
    const char *mode_str,
    void *size,
    void *scale_factor,
    int align_corners) {

    auto x_shape = x_desc->shape();
    auto y_shape = y_desc->shape();

    if (x_shape.size() < 3) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    if ((size != nullptr) == (scale_factor != nullptr)) {
        return INFINI_STATUS_BAD_PARAM;
    }

    if (y_shape.size() != x_shape.size() || y_shape[0] != x_shape[0] || y_shape[1] != x_shape[1]) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    size_t ndim = x_shape.size() - 2;

    if (scale_factor != nullptr) {
        const double *scale_array = reinterpret_cast<const double *>(scale_factor);
        const double scale = scale_array[0];
        std::vector<size_t> expected_y_shape = x_shape;
        for (size_t i = 0; i < ndim; ++i) {
            expected_y_shape[i + 2] = static_cast<size_t>(static_cast<double>(x_shape[i + 2]) * scale);
        }
        if (y_shape != expected_y_shape) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
    }

    InterpolateInfo info;
    info.ndim = ndim;
    info.input_shape = x_shape;
    info.output_shape = y_shape;
    if (!try_parse_mode(mode_str, info.mode)) {
        return INFINI_STATUS_BAD_PARAM;
    }
    info.input_strides = x_desc->strides();
    info.output_strides = y_desc->strides();
    info.align_corners = align_corners;
    info.input_size = x_desc->numel();
    info.output_size = y_desc->numel();

    return utils::Result<InterpolateInfo>(std::move(info));
}

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    const char *mode,
    void *size,
    void *scale_factor,
    int align_corners) {

    auto dtype = x_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_BF16);

    auto info_result = InterpolateInfo::create(x_desc, y_desc, mode, size, scale_factor, align_corners);
    CHECK_RESULT(info_result);

    *desc_ptr = new Descriptor(dtype, info_result.take(), handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename T>
void interpolate_nearest_1d(
    const InterpolateInfo &info,
    const T *input, T *output,
    size_t batch, size_t channels,
    size_t in_w, size_t out_w,
    int align_corners) {

    double scale = compute_scale(in_w, out_w, align_corners);

    const auto &in_s = info.input_strides;
    const auto &out_s = info.output_strides;

    for (size_t b = 0; b < batch; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t ow = 0; ow < out_w; ++ow) {
                double src_x = align_corners ? ow * scale : (ow + 0.5) * scale - 0.5;
                size_t ix = std::min(static_cast<size_t>(std::round(src_x)), in_w - 1);
                output[offset_3d(out_s, b, c, ow)] = input[offset_3d(in_s, b, c, ix)];
            }
        }
    }
}

template <typename T>
void interpolate_bilinear_2d(
    const InterpolateInfo &info,
    const T *input, T *output,
    size_t batch, size_t channels,
    size_t in_h, size_t in_w,
    size_t out_h, size_t out_w,
    int align_corners) {

    double scale_h = compute_scale(in_h, out_h, align_corners);
    double scale_w = compute_scale(in_w, out_w, align_corners);

    const auto &in_s = info.input_strides;
    const auto &out_s = info.output_strides;

    for (size_t b = 0; b < batch; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t oh = 0; oh < out_h; ++oh) {
                for (size_t ow = 0; ow < out_w; ++ow) {
                    double src_y = align_corners ? oh * scale_h : (oh + 0.5) * scale_h - 0.5;
                    double src_x = align_corners ? ow * scale_w : (ow + 0.5) * scale_w - 0.5;

                    src_y = std::max(0.0, std::min(src_y, (double)(in_h - 1)));
                    src_x = std::max(0.0, std::min(src_x, (double)(in_w - 1)));

                    size_t y0 = (size_t)std::floor(src_y);
                    size_t y1 = std::min(y0 + 1, in_h - 1);
                    size_t x0 = (size_t)std::floor(src_x);
                    size_t x1 = std::min(x0 + 1, in_w - 1);

                    double dy = src_y - y0;
                    double dx = src_x - x0;

                    T v00 = input[offset_4d(in_s, b, c, y0, x0)];
                    T v01 = input[offset_4d(in_s, b, c, y0, x1)];
                    T v10 = input[offset_4d(in_s, b, c, y1, x0)];
                    T v11 = input[offset_4d(in_s, b, c, y1, x1)];

                    T result = utils::cast<T>(
                        (1 - dy) * (1 - dx) * utils::cast<double>(v00) + (1 - dy) * dx * utils::cast<double>(v01) + dy * (1 - dx) * utils::cast<double>(v10) + dy * dx * utils::cast<double>(v11));

                    output[offset_4d(out_s, b, c, oh, ow)] = result;
                }
            }
        }
    }
}

template <typename T>
void interpolate_trilinear_3d(
    const InterpolateInfo &info,
    const T *input, T *output,
    size_t batch, size_t channels,
    size_t in_d, size_t in_h, size_t in_w,
    size_t out_d, size_t out_h, size_t out_w,
    int align_corners) {

    double scale_d = compute_scale(in_d, out_d, align_corners);
    double scale_h = compute_scale(in_h, out_h, align_corners);
    double scale_w = compute_scale(in_w, out_w, align_corners);

    const auto &in_s = info.input_strides;
    const auto &out_s = info.output_strides;

    auto offset_5d = [](const std::vector<ptrdiff_t> &s,
                        size_t b, size_t c, size_t d, size_t h, size_t w) {
        return b * s[0] + c * s[1] + d * s[2] + h * s[3] + w * s[4];
    };

    for (size_t b = 0; b < batch; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t od = 0; od < out_d; ++od) {
                for (size_t oh = 0; oh < out_h; ++oh) {
                    for (size_t ow = 0; ow < out_w; ++ow) {

                        double src_d = align_corners ? od * scale_d : (od + 0.5) * scale_d - 0.5;
                        double src_h = align_corners ? oh * scale_h : (oh + 0.5) * scale_h - 0.5;
                        double src_w = align_corners ? ow * scale_w : (ow + 0.5) * scale_w - 0.5;

                        src_d = std::max(0.0, std::min(src_d, (double)(in_d - 1)));
                        src_h = std::max(0.0, std::min(src_h, (double)(in_h - 1)));
                        src_w = std::max(0.0, std::min(src_w, (double)(in_w - 1)));

                        size_t d0 = (size_t)std::floor(src_d);
                        size_t d1 = std::min(d0 + 1, in_d - 1);
                        size_t h0 = (size_t)std::floor(src_h);
                        size_t h1 = std::min(h0 + 1, in_h - 1);
                        size_t w0 = (size_t)std::floor(src_w);
                        size_t w1 = std::min(w0 + 1, in_w - 1);

                        double dd = src_d - d0;
                        double dh = src_h - h0;
                        double dw = src_w - w0;

                        // 8 corners
                        T c000 = input[offset_5d(in_s, b, c, d0, h0, w0)];
                        T c001 = input[offset_5d(in_s, b, c, d0, h0, w1)];
                        T c010 = input[offset_5d(in_s, b, c, d0, h1, w0)];
                        T c011 = input[offset_5d(in_s, b, c, d0, h1, w1)];
                        T c100 = input[offset_5d(in_s, b, c, d1, h0, w0)];
                        T c101 = input[offset_5d(in_s, b, c, d1, h0, w1)];
                        T c110 = input[offset_5d(in_s, b, c, d1, h1, w0)];
                        T c111 = input[offset_5d(in_s, b, c, d1, h1, w1)];

                        // trilinear interpolation
                        double v = (1 - dd) * (1 - dh) * (1 - dw) * utils::cast<double>(c000) + (1 - dd) * (1 - dh) * dw * utils::cast<double>(c001) + (1 - dd) * dh * (1 - dw) * utils::cast<double>(c010) + (1 - dd) * dh * dw * utils::cast<double>(c011) + dd * (1 - dh) * (1 - dw) * utils::cast<double>(c100) + dd * (1 - dh) * dw * utils::cast<double>(c101) + dd * dh * (1 - dw) * utils::cast<double>(c110) + dd * dh * dw * utils::cast<double>(c111);

                        output[offset_5d(out_s, b, c, od, oh, ow)] = utils::cast<T>(v);
                    }
                }
            }
        }
    }
}

template <typename T>
void interpolate_impl(const InterpolateInfo &info, T *y, const T *x) {

    size_t batch = info.input_shape[0];
    size_t channels = info.input_shape[1];

    const auto &in_s = info.input_strides;
    const auto &out_s = info.output_strides;

    if (info.mode == InterpolateMode::NEAREST) {
        if (info.ndim == 1) {
            interpolate_nearest_1d(info, x, y, batch, channels,
                                   info.input_shape[2], info.output_shape[2],
                                   info.align_corners);
        } else if (info.ndim == 2) {
            size_t in_h = info.input_shape[2];
            size_t in_w = info.input_shape[3];
            size_t out_h = info.output_shape[2];
            size_t out_w = info.output_shape[3];

            double scale_h = compute_scale(in_h, out_h, info.align_corners);
            double scale_w = compute_scale(in_w, out_w, info.align_corners);

            for (size_t b = 0; b < batch; ++b) {
                for (size_t c = 0; c < channels; ++c) {
                    for (size_t oh = 0; oh < out_h; ++oh) {
                        for (size_t ow = 0; ow < out_w; ++ow) {
                            double src_y = info.align_corners ? oh * scale_h : (oh + 0.5) * scale_h - 0.5;
                            double src_x = info.align_corners ? ow * scale_w : (ow + 0.5) * scale_w - 0.5;

                            size_t iy = std::min((size_t)std::round(src_y), in_h - 1);
                            size_t ix = std::min((size_t)std::round(src_x), in_w - 1);

                            y[offset_4d(out_s, b, c, oh, ow)] = x[offset_4d(in_s, b, c, iy, ix)];
                        }
                    }
                }
            }
        }
    } else if (info.mode == InterpolateMode::LINEAR || info.mode == InterpolateMode::BILINEAR || info.mode == InterpolateMode::TRILINEAR) {
        if (info.ndim == 1) {
            size_t in_w = info.input_shape[2];
            size_t out_w = info.output_shape[2];
            double scale = compute_scale(in_w, out_w, info.align_corners);

            for (size_t b = 0; b < batch; ++b) {
                for (size_t c = 0; c < channels; ++c) {
                    for (size_t ow = 0; ow < out_w; ++ow) {
                        double src_x = info.align_corners ? ow * scale : (ow + 0.5) * scale - 0.5;
                        src_x = std::max(0.0, std::min(src_x, (double)(in_w - 1)));

                        size_t x0 = (size_t)std::floor(src_x);
                        size_t x1 = std::min(x0 + 1, in_w - 1);

                        double dx = src_x - x0;

                        T v0 = x[offset_3d(in_s, b, c, x0)];
                        T v1 = x[offset_3d(in_s, b, c, x1)];

                        y[offset_3d(out_s, b, c, ow)] = utils::cast<T>((1 - dx) * utils::cast<double>(v0) + dx * utils::cast<double>(v1));
                    }
                }
            }
        } else if (info.ndim == 2) {
            interpolate_bilinear_2d(info, x, y, batch, channels,
                                    info.input_shape[2], info.input_shape[3],
                                    info.output_shape[2], info.output_shape[3],
                                    info.align_corners);
        } else if (info.ndim == 3) {
            size_t in_d = info.input_shape[2];
            size_t in_h = info.input_shape[3];
            size_t in_w = info.input_shape[4];
            size_t out_d = info.output_shape[2];
            size_t out_h = info.output_shape[3];
            size_t out_w = info.output_shape[4];

            interpolate_trilinear_3d(info, x, y, batch, channels,
                                     in_d, in_h, in_w,
                                     out_d, out_h, out_w,
                                     info.align_corners);
        }
    } else if (info.mode == InterpolateMode::AREA) {
        size_t in_h = info.input_shape[2];
        size_t in_w = info.input_shape[3];
        size_t out_h = info.output_shape[2];
        size_t out_w = info.output_shape[3];

        double scale_h = (double)in_h / out_h;
        double scale_w = (double)in_w / out_w;

        for (size_t b = 0; b < batch; ++b) {
            for (size_t c = 0; c < channels; ++c) {
                for (size_t oh = 0; oh < out_h; ++oh) {
                    for (size_t ow = 0; ow < out_w; ++ow) {

                        double start_h = oh * scale_h;
                        double end_h = (oh + 1) * scale_h;
                        double start_w = ow * scale_w;
                        double end_w = (ow + 1) * scale_w;

                        size_t h0 = (size_t)std::floor(start_h);
                        size_t h1 = (size_t)std::ceil(end_h);
                        size_t w0 = (size_t)std::floor(start_w);
                        size_t w1 = (size_t)std::ceil(end_w);

                        double sum = 0.0;
                        size_t count = 0;

                        for (size_t ih = h0; ih < h1 && ih < in_h; ++ih) {
                            for (size_t iw = w0; iw < w1 && iw < in_w; ++iw) {
                                sum += utils::cast<double>(
                                    x[offset_4d(in_s, b, c, ih, iw)]);
                                count++;
                            }
                        }

                        y[offset_4d(out_s, b, c, oh, ow)] = count > 0 ? utils::cast<T>(sum / (double)count)
                                                                      : utils::cast<T>(0.0);
                    }
                }
            }
        }
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
        interpolate_impl<fp16_t>(_info, reinterpret_cast<fp16_t *>(y), reinterpret_cast<const fp16_t *>(x));
        break;
    case INFINI_DTYPE_BF16:
        interpolate_impl<bf16_t>(_info, reinterpret_cast<bf16_t *>(y), reinterpret_cast<const bf16_t *>(x));
        break;
    case INFINI_DTYPE_F32:
        interpolate_impl<float>(_info, reinterpret_cast<float *>(y), reinterpret_cast<const float *>(x));
        break;
    case INFINI_DTYPE_F64:
        interpolate_impl<double>(_info, reinterpret_cast<double *>(y), reinterpret_cast<const double *>(x));
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::interpolate::cpu
