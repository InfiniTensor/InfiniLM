#include "hyper_connection.hpp"

#include "execution/submit.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace infinilm::models::deepseek_v4 {
namespace {

using runtime::DataType;
using runtime::Runtime;
using runtime::Shape;
using runtime::Tensor;
using namespace infini::ops;

template <class Op, class... Args>
Tensor evaluate(Runtime &rt, const Shape &shape, DataType dtype, const Args &...args) {
    auto out = Tensor::empty(rt, shape, dtype);
    execution::submit<Op>(rt, Op::DefaultConfig(rt.device().type()), args..., out);
    return out;
}

template <class Op, class... Args>
Tensor fp32(Runtime &rt, const Shape &shape, const Args &...args) {
    return evaluate<Op>(rt, shape, DataType::kFloat32, args...);
}

Tensor cast(Runtime &rt, const Tensor &x, DataType dtype) {
    return x->dtype() == dtype ? x : evaluate<Cast>(rt, x->shape(), dtype, x);
}

Tensor scalar(Runtime &rt, double value) {
    auto out = Tensor::empty(rt, {1}, DataType::kFloat32);
    execution::submit<Fill>(rt, Fill::DefaultConfig(rt.device().type()), out, value, out);
    return out;
}

Tensor axis_slice(const Tensor &x, size_t axis, size_t start, size_t length) {
    return x->narrow({{axis, start, length}});
}

Tensor reduce_sum(Runtime &rt, const Tensor &x, int64_t axis, const Shape &shape, bool keep_dim) {
    return fp32<Sum>(rt, shape, x, std::optional{std::vector<int64_t>{axis}},
                     keep_dim, std::optional<DataType>{});
}

void require_fp32(const Tensor &tensor, const Shape &shape) {
    if (!tensor || tensor->dtype() != DataType::kFloat32 || tensor->shape() != shape) {
        throw std::invalid_argument("DeepSeek V4 mHC weight shape or dtype mismatch");
    }
}

void validate_residual(const Tensor &residual) {
    if (!residual || residual->ndim() != 3 || !residual->numel() || !residual->is_contiguous()
        || (residual->dtype() != DataType::kFloat32 && residual->dtype() != DataType::kBFloat16)) {
        throw std::invalid_argument("DeepSeek V4 mHC requires a contiguous FP32 or BF16 residual");
    }
}

Tensor project(Runtime &rt, const Tensor &flat, const Tensor &projection,
               size_t outputs, double norm_epsilon) {
    const auto tokens = flat->size(0);
    auto squared = fp32<Mul>(rt, flat->shape(), flat, flat);
    auto mean = fp32<Mean>(rt, {tokens, 1}, squared, std::optional{std::vector<int64_t>{1}},
                           true, std::optional<DataType>{});
    auto variance = fp32<Add>(rt, {tokens, 1}, mean, scalar(rt, norm_epsilon), 1.0);
    auto inv_norm = fp32<Rsqrt>(rt, {tokens, 1}, variance);
    auto projected = Tensor::empty(rt, {tokens, outputs}, DataType::kFloat32);
    // The Hygon provider already computes FP32 via CUBLAS_COMPUTE_32F.
    // NVIDIA's provider needs an explicit policy to avoid reduced precision.
    const GemmConfig full{GemmConfig::Fp32Precision::FullFp32};
    const auto fallback = Gemm::DefaultConfig(rt.device().type());
    const Config &config = rt.device().type() == runtime::Device::Type::kNvidia
                              ? static_cast<const Config &>(full) : fallback;
    execution::submit<Gemm>(rt, config, flat, projection, std::optional<Tensor>{},
                           std::optional<float>{}, std::optional<float>{}, std::optional{0},
                           std::optional{1}, projected);
    return fp32<Mul>(rt, {tokens, outputs}, projected, inv_norm);
}

Tensor affine(Runtime &rt, const Tensor &x, const Tensor &scale, const Tensor &bias) {
    auto scaled = fp32<Mul>(rt, x->shape(), x, scale);
    return fp32<Add>(rt, x->shape(), scaled, bias, 1.0);
}

Tensor collapse(Runtime &rt, const Tensor &residual, const Tensor &pre, DataType dtype) {
    auto weighted = fp32<Mul>(rt, residual->shape(), residual, pre->unsqueeze(2));
    return cast(rt, reduce_sum(rt, weighted, 1, {residual->size(0), residual->size(2)}, false), dtype);
}

void validate_epsilon(double norm_epsilon, double mix_epsilon) {
    if (!std::isfinite(norm_epsilon) || !std::isfinite(mix_epsilon)
        || norm_epsilon <= 0 || mix_epsilon <= 0) {
        throw std::invalid_argument("DeepSeek V4 mHC requires positive finite epsilon values");
    }
}

} // namespace

HyperConnectionMix hyper_connection_pre(
    Runtime &rt, const Tensor &residual, const Tensor &projection, const Tensor &scale,
    const Tensor &bias, double norm_epsilon, double mix_epsilon, size_t sinkhorn_iterations) {
    validate_residual(residual);
    validate_epsilon(norm_epsilon, mix_epsilon);
    const auto tokens = residual->size(0);
    const auto streams = residual->size(1);
    const auto hidden = residual->size(2);
    if (!sinkhorn_iterations || streams > 16) {
        throw std::invalid_argument("DeepSeek V4 mHC requires nonzero Sinkhorn iterations and at most 16 streams");
    }
    const auto outputs = (2 + streams) * streams;
    require_fp32(projection, {outputs, streams * hidden});
    require_fp32(scale, {3});
    require_fp32(bias, {outputs});

    auto x = cast(rt, residual, DataType::kFloat32);
    auto projected = project(rt, x->view({tokens, streams * hidden}), projection, outputs, norm_epsilon);
    auto mix_slice = [&](size_t start, size_t length, size_t scale_index) {
        return affine(rt, axis_slice(projected, 1, start, length),
                      axis_slice(scale, 0, scale_index, 1), axis_slice(bias, 0, start, length));
    };
    const Shape coefficients{tokens, streams};
    auto eps = scalar(rt, mix_epsilon);
    auto pre = fp32<Sigmoid>(rt, coefficients, mix_slice(0, streams, 0));
    pre = fp32<Add>(rt, coefficients, pre, eps, 1.0);
    auto post = fp32<Sigmoid>(rt, coefficients, mix_slice(streams, streams, 1));
    post = fp32<Mul>(rt, coefficients, post, scalar(rt, 2.0));

    const Shape matrix_shape{tokens, streams, streams};
    auto matrix = mix_slice(2 * streams, streams * streams, 2)->view(matrix_shape);
    matrix = fp32<Softmax>(rt, matrix_shape, matrix, int64_t{2}, std::optional<DataType>{});
    matrix = fp32<Add>(rt, matrix_shape, matrix, eps, 1.0);
    const auto normalize = [&](const Tensor &m, int64_t axis, const Shape &shape) {
        auto total = reduce_sum(rt, m, axis, shape, true);
        total = fp32<Add>(rt, shape, total, eps, 1.0);
        return fp32<Div>(rt, matrix_shape, m, total);
    };
    matrix = normalize(matrix, 1, {tokens, 1, streams});
    for (size_t iteration = 1; iteration < sinkhorn_iterations; ++iteration) {
        matrix = normalize(matrix, 2, {tokens, streams, 1});
        matrix = normalize(matrix, 1, {tokens, 1, streams});
    }
    return {collapse(rt, x, pre, residual->dtype()), post, matrix};
}

Tensor hyper_connection_post(Runtime &rt, const Tensor &activation, const Tensor &residual,
                             const HyperConnectionMix &mix) {
    validate_residual(residual);
    const auto tokens = residual->size(0);
    const auto streams = residual->size(1);
    const auto hidden = residual->size(2);
    if (!activation || activation->shape() != Shape{tokens, hidden}
        || activation->dtype() != residual->dtype()) {
        throw std::invalid_argument("DeepSeek V4 mHC activation shape or dtype mismatch");
    }
    require_fp32(mix.post, {tokens, streams});
    require_fp32(mix.combination, {tokens, streams, streams});
    auto x = cast(rt, activation, DataType::kFloat32);
    auto r = cast(rt, residual, DataType::kFloat32);
    auto expanded = fp32<Mul>(rt, residual->shape(), x->unsqueeze(1), mix.post->unsqueeze(2));
    auto terms = fp32<Mul>(rt, {tokens, streams, streams, hidden},
                          mix.combination->unsqueeze(3), r->unsqueeze(2));
    auto combined = reduce_sum(rt, terms, 1, residual->shape(), false);
    return cast(rt, fp32<Add>(rt, residual->shape(), expanded, combined, 1.0), activation->dtype());
}

Tensor hyper_connection_head(Runtime &rt, const Tensor &residual, const Tensor &projection,
                             const Tensor &scale, const Tensor &bias,
                             double norm_epsilon, double mix_epsilon) {
    validate_residual(residual);
    validate_epsilon(norm_epsilon, mix_epsilon);
    const auto tokens = residual->size(0);
    const auto streams = residual->size(1);
    const auto hidden = residual->size(2);
    require_fp32(projection, {streams, streams * hidden});
    require_fp32(scale, {1});
    require_fp32(bias, {streams});
    auto x = cast(rt, residual, DataType::kFloat32);
    auto projected = project(rt, x->view({tokens, streams * hidden}), projection, streams, norm_epsilon);
    auto pre = fp32<Sigmoid>(rt, {tokens, streams}, affine(rt, projected, scale, bias));
    pre = fp32<Add>(rt, {tokens, streams}, pre, scalar(rt, mix_epsilon), 1.0);
    return collapse(rt, x, pre, residual->dtype());
}

} // namespace infinilm::models::deepseek_v4
