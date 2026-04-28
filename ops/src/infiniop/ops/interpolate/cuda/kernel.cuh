#pragma once
#include <cmath>
#include <type_traits>

namespace op::interpolate::cuda {

constexpr int kInterpolateMaxDims = 8;

struct TensorMeta {
    int ndim;
    size_t shape[kInterpolateMaxDims];
    ptrdiff_t strides[kInterpolateMaxDims]; // strides in elements
};

template <typename T>
struct TypeTag {};

template <typename Tcompute>
__device__ __forceinline__ Tcompute to_compute(const half v) {
    return static_cast<Tcompute>(__half2float(v));
}

template <typename Tcompute>
__device__ __forceinline__ Tcompute to_compute(const cuda_bfloat16 v) {
    return static_cast<Tcompute>(__bfloat162float(v));
}

template <typename Tcompute, typename T>
__device__ __forceinline__ Tcompute to_compute(const T v) {
    return static_cast<Tcompute>(v);
}

__device__ __forceinline__ half from_compute(const float v, TypeTag<half>) {
    return __float2half_rn(v);
}

__device__ __forceinline__ cuda_bfloat16 from_compute(const float v, TypeTag<cuda_bfloat16>) {
    return __float2bfloat16_rn(v);
}

template <typename Tcompute, typename T>
__device__ __forceinline__ T from_compute(const Tcompute v, TypeTag<T>) {
    return static_cast<T>(v);
}

__device__ __forceinline__ double compute_scale(size_t in_size, size_t out_size, bool align_corners) {
    if (out_size == 0) {
        return 0.0;
    }
    if (align_corners) {
        if (out_size == 1) {
            return 0.0;
        }
        return (static_cast<double>(in_size) - 1.0) / (static_cast<double>(out_size) - 1.0);
    }
    return static_cast<double>(in_size) / static_cast<double>(out_size);
}

__device__ __forceinline__ double compute_source_index(
    int64_t out_idx,
    double scale,
    bool align_corners) {
    if (align_corners) {
        return static_cast<double>(out_idx) * scale;
    }
    return (static_cast<double>(out_idx) + 0.5) * scale - 0.5;
}

template <typename T>
__global__ void nearest_2d_kernel(
    T *output,
    const T *input,
    TensorMeta out_meta,
    TensorMeta in_meta) {

    const size_t total = out_meta.shape[0] * out_meta.shape[1] * out_meta.shape[2] * out_meta.shape[3];
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    const size_t out_w = out_meta.shape[3];
    const size_t out_h = out_meta.shape[2];
    const size_t c_stride = out_h * out_w;
    const size_t n_stride = out_meta.shape[1] * c_stride;

    const size_t n = idx / n_stride;
    size_t rem = idx - n * n_stride;
    const size_t c = rem / c_stride;
    rem -= c * c_stride;
    const size_t oh = rem / out_w;
    const size_t ow = rem - oh * out_w;

    const size_t in_h = in_meta.shape[2];
    const size_t in_w = in_meta.shape[3];

    const double scale_h = static_cast<double>(in_h) / static_cast<double>(out_h);
    const double scale_w = static_cast<double>(in_w) / static_cast<double>(out_w);

    const size_t ih_raw = static_cast<size_t>(floor(static_cast<double>(oh) * scale_h));
    const size_t iw_raw = static_cast<size_t>(floor(static_cast<double>(ow) * scale_w));
    const size_t ih = (ih_raw < in_h) ? ih_raw : (in_h - 1);
    const size_t iw = (iw_raw < in_w) ? iw_raw : (in_w - 1);

    const size_t in_off = n * in_meta.strides[0] + c * in_meta.strides[1] + ih * in_meta.strides[2] + iw * in_meta.strides[3];
    const size_t out_off = n * out_meta.strides[0] + c * out_meta.strides[1] + oh * out_meta.strides[2] + ow * out_meta.strides[3];

    output[out_off] = input[in_off];
}

template <typename T, typename Tcompute>
__global__ void linear_1d_kernel(
    T *output,
    const T *input,
    TensorMeta out_meta,
    TensorMeta in_meta,
    int align_corners) {

    const size_t total = out_meta.shape[0] * out_meta.shape[1] * out_meta.shape[2];
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    const size_t out_w = out_meta.shape[2];
    const size_t c_stride = out_w;
    const size_t n_stride = out_meta.shape[1] * c_stride;

    const size_t n = idx / n_stride;
    size_t rem = idx - n * n_stride;
    const size_t c = rem / c_stride;
    const size_t ow = rem - c * c_stride;

    const size_t in_w = in_meta.shape[2];
    const bool use_align = (align_corners != 0);
    const double scale = compute_scale(in_w, out_w, use_align);

    double src = compute_source_index(static_cast<int64_t>(ow), scale, use_align);
    src = fmax(0.0, fmin(src, static_cast<double>(in_w - 1)));

    const size_t x0 = static_cast<size_t>(floor(src));
    const size_t x1 = (x0 + 1 < in_w) ? (x0 + 1) : x0;
    const double t = src - static_cast<double>(x0);

    const size_t off0 = n * in_meta.strides[0] + c * in_meta.strides[1] + x0 * in_meta.strides[2];
    const size_t off1 = n * in_meta.strides[0] + c * in_meta.strides[1] + x1 * in_meta.strides[2];
    const Tcompute v0 = to_compute<Tcompute>(input[off0]);
    const Tcompute v1 = to_compute<Tcompute>(input[off1]);
    const Tcompute out = static_cast<Tcompute>((1.0 - t) * static_cast<double>(v0) + t * static_cast<double>(v1));

    const size_t out_off = n * out_meta.strides[0] + c * out_meta.strides[1] + ow * out_meta.strides[2];
    output[out_off] = from_compute(out, TypeTag<T>{});
}

template <typename T, typename Tcompute>
__global__ void bilinear_2d_kernel(
    T *output,
    const T *input,
    TensorMeta out_meta,
    TensorMeta in_meta,
    int align_corners) {

    const size_t total = out_meta.shape[0] * out_meta.shape[1] * out_meta.shape[2] * out_meta.shape[3];
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    const size_t out_w = out_meta.shape[3];
    const size_t out_h = out_meta.shape[2];
    const size_t c_stride = out_h * out_w;
    const size_t n_stride = out_meta.shape[1] * c_stride;

    const size_t n = idx / n_stride;
    size_t rem = idx - n * n_stride;
    const size_t c = rem / c_stride;
    rem -= c * c_stride;
    const size_t oh = rem / out_w;
    const size_t ow = rem - oh * out_w;

    const size_t in_h = in_meta.shape[2];
    const size_t in_w = in_meta.shape[3];
    const bool use_align = (align_corners != 0);
    const double scale_h = compute_scale(in_h, out_h, use_align);
    const double scale_w = compute_scale(in_w, out_w, use_align);

    double src_y = compute_source_index(static_cast<int64_t>(oh), scale_h, use_align);
    double src_x = compute_source_index(static_cast<int64_t>(ow), scale_w, use_align);
    src_y = fmax(0.0, fmin(src_y, static_cast<double>(in_h - 1)));
    src_x = fmax(0.0, fmin(src_x, static_cast<double>(in_w - 1)));

    const size_t y0 = static_cast<size_t>(floor(src_y));
    const size_t x0 = static_cast<size_t>(floor(src_x));
    const size_t y1 = (y0 + 1 < in_h) ? (y0 + 1) : y0;
    const size_t x1 = (x0 + 1 < in_w) ? (x0 + 1) : x0;
    const double wy = src_y - static_cast<double>(y0);
    const double wx = src_x - static_cast<double>(x0);

    const size_t base = n * in_meta.strides[0] + c * in_meta.strides[1];
    const Tcompute v00 = to_compute<Tcompute>(input[base + y0 * in_meta.strides[2] + x0 * in_meta.strides[3]]);
    const Tcompute v01 = to_compute<Tcompute>(input[base + y0 * in_meta.strides[2] + x1 * in_meta.strides[3]]);
    const Tcompute v10 = to_compute<Tcompute>(input[base + y1 * in_meta.strides[2] + x0 * in_meta.strides[3]]);
    const Tcompute v11 = to_compute<Tcompute>(input[base + y1 * in_meta.strides[2] + x1 * in_meta.strides[3]]);

    const double w00 = (1.0 - wy) * (1.0 - wx);
    const double w01 = (1.0 - wy) * wx;
    const double w10 = wy * (1.0 - wx);
    const double w11 = wy * wx;

    const Tcompute out = static_cast<Tcompute>(
        w00 * static_cast<double>(v00) + w01 * static_cast<double>(v01) + w10 * static_cast<double>(v10) + w11 * static_cast<double>(v11));

    const size_t out_off = n * out_meta.strides[0] + c * out_meta.strides[1] + oh * out_meta.strides[2] + ow * out_meta.strides[3];
    output[out_off] = from_compute(out, TypeTag<T>{});
}

template <typename T, typename Tcompute>
__global__ void trilinear_3d_kernel(
    T *output,
    const T *input,
    TensorMeta out_meta,
    TensorMeta in_meta,
    int align_corners) {

    const size_t total = out_meta.shape[0] * out_meta.shape[1] * out_meta.shape[2] * out_meta.shape[3] * out_meta.shape[4];
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    const size_t out_d = out_meta.shape[2];
    const size_t out_h = out_meta.shape[3];
    const size_t out_w = out_meta.shape[4];
    const size_t c_stride = out_d * out_h * out_w;
    const size_t n_stride = out_meta.shape[1] * c_stride;

    const size_t n = idx / n_stride;
    size_t rem = idx - n * n_stride;
    const size_t c = rem / c_stride;
    rem -= c * c_stride;

    const size_t od = rem / (out_h * out_w);
    rem -= od * (out_h * out_w);
    const size_t oh = rem / out_w;
    const size_t ow = rem - oh * out_w;

    const size_t in_d = in_meta.shape[2];
    const size_t in_h = in_meta.shape[3];
    const size_t in_w = in_meta.shape[4];
    const bool use_align = (align_corners != 0);
    const double scale_d = compute_scale(in_d, out_d, use_align);
    const double scale_h = compute_scale(in_h, out_h, use_align);
    const double scale_w = compute_scale(in_w, out_w, use_align);

    double src_d = compute_source_index(static_cast<int64_t>(od), scale_d, use_align);
    double src_h = compute_source_index(static_cast<int64_t>(oh), scale_h, use_align);
    double src_w = compute_source_index(static_cast<int64_t>(ow), scale_w, use_align);

    src_d = fmax(0.0, fmin(src_d, static_cast<double>(in_d - 1)));
    src_h = fmax(0.0, fmin(src_h, static_cast<double>(in_h - 1)));
    src_w = fmax(0.0, fmin(src_w, static_cast<double>(in_w - 1)));

    const size_t d0 = static_cast<size_t>(floor(src_d));
    const size_t h0 = static_cast<size_t>(floor(src_h));
    const size_t w0 = static_cast<size_t>(floor(src_w));
    const size_t d1 = (d0 + 1 < in_d) ? (d0 + 1) : d0;
    const size_t h1 = (h0 + 1 < in_h) ? (h0 + 1) : h0;
    const size_t w1 = (w0 + 1 < in_w) ? (w0 + 1) : w0;

    const double td = src_d - static_cast<double>(d0);
    const double th = src_h - static_cast<double>(h0);
    const double tw = src_w - static_cast<double>(w0);

    const size_t base = n * in_meta.strides[0] + c * in_meta.strides[1];
    auto load = [&](size_t d, size_t h, size_t w) -> Tcompute {
        return to_compute<Tcompute>(input[base + d * in_meta.strides[2] + h * in_meta.strides[3] + w * in_meta.strides[4]]);
    };

    const Tcompute v000 = load(d0, h0, w0);
    const Tcompute v001 = load(d0, h0, w1);
    const Tcompute v010 = load(d0, h1, w0);
    const Tcompute v011 = load(d0, h1, w1);
    const Tcompute v100 = load(d1, h0, w0);
    const Tcompute v101 = load(d1, h0, w1);
    const Tcompute v110 = load(d1, h1, w0);
    const Tcompute v111 = load(d1, h1, w1);

    const double w000 = (1.0 - td) * (1.0 - th) * (1.0 - tw);
    const double w001 = (1.0 - td) * (1.0 - th) * tw;
    const double w010 = (1.0 - td) * th * (1.0 - tw);
    const double w011 = (1.0 - td) * th * tw;
    const double w100 = td * (1.0 - th) * (1.0 - tw);
    const double w101 = td * (1.0 - th) * tw;
    const double w110 = td * th * (1.0 - tw);
    const double w111 = td * th * tw;

    const Tcompute out = static_cast<Tcompute>(
        w000 * static_cast<double>(v000) + w001 * static_cast<double>(v001) + w010 * static_cast<double>(v010) + w011 * static_cast<double>(v011) + w100 * static_cast<double>(v100) + w101 * static_cast<double>(v101) + w110 * static_cast<double>(v110) + w111 * static_cast<double>(v111));

    const size_t out_off = n * out_meta.strides[0] + c * out_meta.strides[1] + od * out_meta.strides[2] + oh * out_meta.strides[3] + ow * out_meta.strides[4];
    output[out_off] = from_compute(out, TypeTag<T>{});
}

template <typename T, typename Tcompute>
__global__ void area_2d_kernel(
    T *output,
    const T *input,
    TensorMeta out_meta,
    TensorMeta in_meta) {

    const size_t total = out_meta.shape[0] * out_meta.shape[1] * out_meta.shape[2] * out_meta.shape[3];
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    const size_t out_w = out_meta.shape[3];
    const size_t out_h = out_meta.shape[2];
    const size_t c_stride = out_h * out_w;
    const size_t n_stride = out_meta.shape[1] * c_stride;

    const size_t n = idx / n_stride;
    size_t rem = idx - n * n_stride;
    const size_t c = rem / c_stride;
    rem -= c * c_stride;
    const size_t oh = rem / out_w;
    const size_t ow = rem - oh * out_w;

    const size_t in_h = in_meta.shape[2];
    const size_t in_w = in_meta.shape[3];

    const double scale_h = static_cast<double>(in_h) / static_cast<double>(out_h);
    const double scale_w = static_cast<double>(in_w) / static_cast<double>(out_w);

    const double h0 = static_cast<double>(oh) * scale_h;
    const double h1 = static_cast<double>(oh + 1) * scale_h;
    const double w0 = static_cast<double>(ow) * scale_w;
    const double w1 = static_cast<double>(ow + 1) * scale_w;

    const size_t ih0 = static_cast<size_t>(floor(h0));
    const size_t ih1 = static_cast<size_t>(ceil(h1));
    const size_t iw0 = static_cast<size_t>(floor(w0));
    const size_t iw1 = static_cast<size_t>(ceil(w1));

    const size_t base = n * in_meta.strides[0] + c * in_meta.strides[1];

    double accum = 0.0;
    double area = (h1 - h0) * (w1 - w0);
    if (area <= 0.0) {
        area = 1.0;
    }

    for (size_t ih = ih0; ih < ih1 && ih < in_h; ++ih) {
        const double wh = fmax(0.0, fmin(h1, static_cast<double>(ih + 1)) - fmax(h0, static_cast<double>(ih)));
        for (size_t iw = iw0; iw < iw1 && iw < in_w; ++iw) {
            const double ww = fmax(0.0, fmin(w1, static_cast<double>(iw + 1)) - fmax(w0, static_cast<double>(iw)));
            const double w = wh * ww;
            const Tcompute v = to_compute<Tcompute>(input[base + ih * in_meta.strides[2] + iw * in_meta.strides[3]]);
            accum += w * static_cast<double>(v);
        }
    }

    const Tcompute out = static_cast<Tcompute>(accum / area);
    const size_t out_off = n * out_meta.strides[0] + c * out_meta.strides[1] + oh * out_meta.strides[2] + ow * out_meta.strides[3];
    output[out_off] = from_compute(out, TypeTag<T>{});
}

} // namespace op::interpolate::cuda

// ---------------------------------------------------------------------------
// Compatibility wrappers for backends that still reference `op::cuda::*`.
// These assume contiguous NCHW layout.
// ---------------------------------------------------------------------------

namespace op::cuda {

template <typename T>
__global__ void interpolate_bilinear_2d_kernel(
    T *output,
    const T *input,
    size_t batch,
    size_t channels,
    size_t in_h,
    size_t in_w,
    size_t out_h,
    size_t out_w,
    double scale_h,
    double scale_w,
    int align_corners) {

    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = batch * channels * out_h * out_w;
    if (idx >= total) {
        return;
    }

    const size_t n = idx / (channels * out_h * out_w);
    size_t rem = idx - n * (channels * out_h * out_w);
    const size_t c = rem / (out_h * out_w);
    rem -= c * (out_h * out_w);
    const size_t oh = rem / out_w;
    const size_t ow = rem - oh * out_w;

    double src_y = align_corners ? static_cast<double>(oh) * scale_h : (static_cast<double>(oh) + 0.5) * scale_h - 0.5;
    double src_x = align_corners ? static_cast<double>(ow) * scale_w : (static_cast<double>(ow) + 0.5) * scale_w - 0.5;

    src_y = fmax(0.0, fmin(src_y, static_cast<double>(in_h - 1)));
    src_x = fmax(0.0, fmin(src_x, static_cast<double>(in_w - 1)));

    const size_t y0 = static_cast<size_t>(floor(src_y));
    const size_t x0 = static_cast<size_t>(floor(src_x));
    const size_t y1 = (y0 + 1 < in_h) ? (y0 + 1) : y0;
    const size_t x1 = (x0 + 1 < in_w) ? (x0 + 1) : x0;
    const double wy = src_y - static_cast<double>(y0);
    const double wx = src_x - static_cast<double>(x0);

    using Tcompute = std::conditional_t<std::is_same_v<T, double>, double, float>;

    const size_t base = ((n * channels + c) * in_h) * in_w;
    const Tcompute v00 = op::interpolate::cuda::to_compute<Tcompute>(input[base + y0 * in_w + x0]);
    const Tcompute v01 = op::interpolate::cuda::to_compute<Tcompute>(input[base + y0 * in_w + x1]);
    const Tcompute v10 = op::interpolate::cuda::to_compute<Tcompute>(input[base + y1 * in_w + x0]);
    const Tcompute v11 = op::interpolate::cuda::to_compute<Tcompute>(input[base + y1 * in_w + x1]);

    const double w00 = (1.0 - wy) * (1.0 - wx);
    const double w01 = (1.0 - wy) * wx;
    const double w10 = wy * (1.0 - wx);
    const double w11 = wy * wx;

    const Tcompute out = static_cast<Tcompute>(
        w00 * static_cast<double>(v00) + w01 * static_cast<double>(v01) + w10 * static_cast<double>(v10) + w11 * static_cast<double>(v11));

    output[((n * channels + c) * out_h + oh) * out_w + ow] = op::interpolate::cuda::from_compute(out, op::interpolate::cuda::TypeTag<T>{});
}

} // namespace op::cuda
