#include "deepseek_v4_utils.hpp"

#include "../../utils.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/ops/add.hpp"
#include "infinicore/ops/matmul.hpp"
#include "infinicore/ops/rms_norm.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>

namespace infinilm::models::deepseek_v4 {
namespace {

size_t numel_from_shape(const infinicore::Shape &shape) {
    return std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1), std::multiplies<size_t>());
}

float read_float_at(const std::byte *ptr, infinicore::DataType dtype, size_t idx) {
    switch (dtype) {
    case infinicore::DataType::F32:
        return reinterpret_cast<const float *>(ptr)[idx];
    case infinicore::DataType::F16:
        return f16_to_f32(reinterpret_cast<const uint16_t *>(ptr)[idx]);
    case infinicore::DataType::BF16:
        return bf16_to_f32(reinterpret_cast<const uint16_t *>(ptr)[idx]);
    case infinicore::DataType::I64:
        return static_cast<float>(reinterpret_cast<const int64_t *>(ptr)[idx]);
    case infinicore::DataType::I32:
        return static_cast<float>(reinterpret_cast<const int32_t *>(ptr)[idx]);
    default:
        throw std::runtime_error("DeepseekV4: unsupported tensor dtype for float conversion");
    }
}

void write_float_at(std::byte *ptr, infinicore::DataType dtype, size_t idx, float value) {
    switch (dtype) {
    case infinicore::DataType::F32:
        reinterpret_cast<float *>(ptr)[idx] = value;
        break;
    case infinicore::DataType::F16:
        reinterpret_cast<uint16_t *>(ptr)[idx] = f32_to_f16(value);
        break;
    case infinicore::DataType::BF16:
        reinterpret_cast<uint16_t *>(ptr)[idx] = f32_to_bf16(value);
        break;
    default:
        throw std::runtime_error("DeepseekV4: unsupported tensor dtype for float write");
    }
}

void softmax_with_eps_inplace(std::vector<float> &row, double eps) {
    const float max_value = *std::max_element(row.begin(), row.end());
    double sum = 0.0;
    for (auto &v : row) {
        v = std::exp(v - max_value);
        sum += v;
    }
    for (auto &v : row) {
        v = static_cast<float>(v / sum + eps);
    }
}

std::vector<float> softmax_with_eps(const std::vector<float> &row, double eps) {
    std::vector<float> out = row;
    softmax_with_eps_inplace(out, eps);
    return out;
}

float sigmoid(float value) {
    if (value >= 0.0f) {
        const float z = std::exp(-value);
        return 1.0f / (1.0f + z);
    }
    const float z = std::exp(value);
    return z / (1.0f + z);
}

float silu(float value) {
    return value * sigmoid(value);
}

} // namespace

std::vector<float> tensor_to_float_vector(const infinicore::Tensor &tensor) {
    auto cpu = tensor->contiguous()->to(infinicore::Device::cpu());
    const size_t count = cpu->numel();
    const auto dtype = cpu->dtype();
    const auto *ptr = cpu->data();
    std::vector<float> out(count);
    for (size_t i = 0; i < count; ++i) {
        out[i] = read_float_at(ptr, dtype, i);
    }
    return out;
}

std::vector<int64_t> tensor_to_int64_vector(const infinicore::Tensor &tensor) {
    auto cpu = tensor->contiguous()->to(infinicore::Device::cpu());
    const size_t count = cpu->numel();
    const auto dtype = cpu->dtype();
    const auto *ptr = cpu->data();
    std::vector<int64_t> out(count);
    switch (dtype) {
    case infinicore::DataType::I64:
        std::memcpy(out.data(), ptr, count * sizeof(int64_t));
        break;
    case infinicore::DataType::I32:
        for (size_t i = 0; i < count; ++i) {
            out[i] = reinterpret_cast<const int32_t *>(ptr)[i];
        }
        break;
    default:
        throw std::runtime_error("DeepseekV4: unsupported tensor dtype for int64 conversion");
    }
    return out;
}

bool debug_trace_enabled() {
    const char *value = std::getenv("DEEPSEEK_V4_DEBUG_TRACE");
    return value != nullptr && value[0] != '\0' && std::string(value) != "0";
}

bool debug_trace_layer_enabled(size_t layer_idx) {
    if (!debug_trace_enabled()) {
        return false;
    }
    const char *value = std::getenv("DEEPSEEK_V4_DEBUG_LAYER");
    if (value == nullptr || value[0] == '\0') {
        return true;
    }
    try {
        return std::stoll(value) == static_cast<long long>(layer_idx);
    } catch (...) {
        return false;
    }
}

void debug_trace_layer_tensor(const std::string &name, size_t layer_idx, const infinicore::Tensor &tensor) {
    if (!debug_trace_layer_enabled(layer_idx)) {
        return;
    }
    debug_trace_tensor(name + ".layer" + std::to_string(layer_idx), tensor);
}

void debug_trace_tensor(const std::string &name, const infinicore::Tensor &tensor) {
    if (!debug_trace_enabled()) {
        return;
    }
    const auto values = tensor_to_float_vector(tensor);
    if (values.empty()) {
        spdlog::info("DSV4_TRACE {} shape=[] mean=0 rms=0 max_abs=0 first=0 last=0", name);
        return;
    }
    double sum = 0.0;
    double sum_sq = 0.0;
    float max_abs = 0.0f;
    for (float value : values) {
        sum += value;
        sum_sq += static_cast<double>(value) * value;
        max_abs = std::max(max_abs, std::abs(value));
    }
    std::string shape = "[";
    const auto &dims = tensor->shape();
    for (size_t i = 0; i < dims.size(); ++i) {
        if (i > 0) {
            shape += ",";
        }
        shape += std::to_string(dims[i]);
    }
    shape += "]";
    spdlog::info("DSV4_TRACE {} shape={} mean={:.9g} rms={:.9g} max_abs={:.9g} first={:.9g} last={:.9g}",
                 name, shape, sum / static_cast<double>(values.size()),
                 std::sqrt(sum_sq / static_cast<double>(values.size())),
                 max_abs, values.front(), values.back());
}

infinicore::Tensor float_vector_to_tensor(const std::vector<float> &values,
                                          const infinicore::Shape &shape,
                                          infinicore::DataType dtype,
                                          const infinicore::Device &device) {
    if (values.size() != numel_from_shape(shape)) {
        throw std::runtime_error("DeepseekV4: float_vector_to_tensor shape mismatch");
    }
    auto cpu = infinicore::Tensor::empty(shape, dtype, infinicore::Device::cpu());
    auto *ptr = cpu->data();
    for (size_t i = 0; i < values.size(); ++i) {
        write_float_at(ptr, dtype, i, values[i]);
    }
    auto out = cpu->to(device);
    if (device.getType() != infinicore::Device::Type::CPU) {
        infinicore::context::syncStream();
    }
    return out;
}

infinicore::Tensor unweighted_rms_norm(const infinicore::Tensor &x, double eps) {
    const auto shape = x->shape();
    if (shape.empty()) {
        throw std::runtime_error("DeepseekV4: unweighted_rms_norm expects non-scalar tensor");
    }
    const size_t last_dim = shape.back();
    auto values = tensor_to_float_vector(x->contiguous());
    const size_t num_groups = values.size() / last_dim;
    for (size_t group = 0; group < num_groups; ++group) {
        const size_t offset = group * last_dim;
        double mean_square = 0.0;
        for (size_t d = 0; d < last_dim; ++d) {
            mean_square += static_cast<double>(values[offset + d]) * values[offset + d];
        }
        const float rsqrt = static_cast<float>(1.0 / std::sqrt(mean_square / static_cast<double>(last_dim) + eps));
        for (size_t d = 0; d < last_dim; ++d) {
            values[offset + d] *= rsqrt;
        }
    }
    return float_vector_to_tensor(values, shape, x->dtype(), x->device());
}

std::vector<int64_t> normalize_positions(const infinicore::Tensor &positions, size_t seq_len) {
    auto values = tensor_to_int64_vector(positions);
    if (values.size() == seq_len) {
        return values;
    }
    if (values.size() >= seq_len) {
        return std::vector<int64_t>(values.end() - static_cast<std::ptrdiff_t>(seq_len), values.end());
    }
    std::vector<int64_t> out(seq_len);
    for (size_t i = 0; i < seq_len; ++i) {
        out[i] = static_cast<int64_t>(i);
    }
    return out;
}

void apply_partial_rope_inplace(std::vector<float> &x,
                                size_t offset,
                                size_t head_dim,
                                size_t rope_dim,
                                int64_t position,
                                double theta,
                                bool inverse) {
    if (rope_dim == 0) {
        return;
    }
    const size_t pass_dim = head_dim - rope_dim;
    const size_t half = rope_dim / 2;
    std::vector<float> old(rope_dim);
    for (size_t i = 0; i < rope_dim; ++i) {
        old[i] = x[offset + pass_dim + i];
    }
    for (size_t i = 0; i < half; ++i) {
        const double inv_freq = 1.0 / std::pow(theta, static_cast<double>(2 * i) / static_cast<double>(rope_dim));
        const double angle = static_cast<double>(position) * inv_freq;
        const float c = static_cast<float>(std::cos(angle));
        const float s = static_cast<float>((inverse ? -1.0 : 1.0) * std::sin(angle));
        const size_t even = 2 * i;
        const size_t odd = even + 1;
        const float x1 = old[even];
        const float x2 = old[odd];
        x[offset + pass_dim + even] = x1 * c - x2 * s;
        x[offset + pass_dim + odd] = x2 * c + x1 * s;
    }
}

double deepseek_v4_yarn_inv_freq(size_t pair_idx,
                                 size_t rope_dim,
                                 double base,
                                 double factor,
                                 double beta_fast,
                                 double beta_slow,
                                 int64_t original_seq_len,
                                 double extrapolation_factor) {
    const double inv_freq_extrapolation = 1.0 / std::pow(base, static_cast<double>(2 * pair_idx) / static_cast<double>(rope_dim));
    if (factor <= 1.0 || original_seq_len <= 0 || base <= 1.0) {
        return inv_freq_extrapolation;
    }

    constexpr double pi = 3.141592653589793238462643383279502884;
    auto find_correction_dim = [&](double num_rotations) {
        return static_cast<double>(rope_dim)
             * std::log(static_cast<double>(original_seq_len) / (num_rotations * 2.0 * pi))
             / (2.0 * std::log(base));
    };
    double low = std::floor(find_correction_dim(beta_fast));
    double high = std::ceil(find_correction_dim(beta_slow));
    low = std::max(low, 0.0);
    high = std::min(high, static_cast<double>(rope_dim - 1));
    if (low == high) {
        high += 0.001;
    }

    const double ramp = std::clamp((static_cast<double>(pair_idx) - low) / (high - low), 0.0, 1.0);
    const double inv_freq_mask = (1.0 - ramp) * extrapolation_factor;
    const double inv_freq_interpolation = inv_freq_extrapolation / factor;
    return inv_freq_interpolation * (1.0 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask;
}

void apply_deepseek_v4_yarn_rope_inplace(std::vector<float> &x,
                                         size_t offset,
                                         size_t head_dim,
                                         size_t rope_dim,
                                         int64_t position,
                                         double base,
                                         double factor,
                                         double beta_fast,
                                         double beta_slow,
                                         int64_t original_seq_len,
                                         double extrapolation_factor,
                                         bool inverse) {
    if (rope_dim == 0) {
        return;
    }
    const size_t pass_dim = head_dim - rope_dim;
    const size_t half = rope_dim / 2;
    std::vector<float> old(rope_dim);
    for (size_t i = 0; i < rope_dim; ++i) {
        old[i] = x[offset + pass_dim + i];
    }
    for (size_t i = 0; i < half; ++i) {
        const double inv_freq = deepseek_v4_yarn_inv_freq(
            i, rope_dim, base, factor, beta_fast, beta_slow, original_seq_len, extrapolation_factor);
        const double angle = static_cast<double>(position) * inv_freq;
        const float c = static_cast<float>(std::cos(angle));
        const float s = static_cast<float>((inverse ? -1.0 : 1.0) * std::sin(angle));
        const size_t even = 2 * i;
        const size_t odd = even + 1;
        const float x1 = old[even];
        const float x2 = old[odd];
        x[offset + pass_dim + even] = x1 * c - x2 * s;
        x[offset + pass_dim + odd] = x2 * c + x1 * s;
    }
}

void apply_rope_at_offset(std::vector<float> &values,
                          size_t offset,
                          int64_t position,
                          const DeepseekV4RopeParams &cfg,
                          bool inverse) {
    if (cfg.use_yarn) {
        apply_deepseek_v4_yarn_rope_inplace(values,
                                            offset,
                                            cfg.head_dim,
                                            cfg.rope_dim,
                                            position,
                                            cfg.rope_theta,
                                            cfg.yarn_factor,
                                            cfg.yarn_beta_fast,
                                            cfg.yarn_beta_slow,
                                            cfg.yarn_original_seq_len,
                                            cfg.yarn_extrapolation_factor,
                                            inverse);
    } else {
        apply_partial_rope_inplace(values, offset, cfg.head_dim, cfg.rope_dim, position, cfg.rope_theta, inverse);
    }
}

infinicore::Tensor apply_rotary_pos_emb(const infinicore::Tensor &x,
                                        const std::vector<int64_t> &positions,
                                        const DeepseekV4RopeParams &cfg,
                                        bool inverse) {
    const auto shape = x->shape();
    if (shape.size() != 4) {
        throw std::runtime_error("DeepseekV4: apply_rotary_pos_emb expects [B,S,H,D]");
    }
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    const size_t num_heads = shape[2];
    const size_t head_dim = shape[3];
    if (positions.size() != seq_len) {
        throw std::runtime_error("DeepseekV4: apply_rotary_pos_emb positions length mismatch");
    }
    if (head_dim != cfg.head_dim) {
        throw std::runtime_error("DeepseekV4: apply_rotary_pos_emb head_dim mismatch");
    }

    auto values = tensor_to_float_vector(x->contiguous());
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t t = 0; t < seq_len; ++t) {
            const int64_t position = positions[t];
            for (size_t h = 0; h < num_heads; ++h) {
                const size_t offset = ((b * seq_len + t) * num_heads + h) * head_dim;
                apply_rope_at_offset(values, offset, position, cfg, inverse);
            }
        }
    }
    return float_vector_to_tensor(values, shape, x->dtype(), x->device());
}

infinicore::Tensor int64_vector_to_tensor(const std::vector<int64_t> &values,
                                          const infinicore::Shape &shape,
                                          const infinicore::Device &device) {
    if (values.size() != numel_from_shape(shape)) {
        throw std::runtime_error("DeepseekV4: int64_vector_to_tensor shape mismatch");
    }
    auto cpu = infinicore::Tensor::empty(shape, infinicore::DataType::I64, infinicore::Device::cpu());
    std::memcpy(cpu->data(), values.data(), values.size() * sizeof(int64_t));
    auto out = cpu->to(device);
    if (device.getType() != infinicore::Device::Type::CPU) {
        infinicore::context::syncStream();
    }
    return out;
}

namespace {

void fill_mhc_params_from_mixes(DeepseekV4MHCParams &params,
                                const std::vector<float> &mixes,
                                const DeepseekV4MHCCoeffs &coeffs,
                                size_t sinkhorn_iters,
                                double eps) {
    const size_t batch_size = params.batch_size;
    const size_t seq_len = params.seq_len;
    const size_t hc_mult = params.hc_mult;
    const size_t mix_hc = coeffs.mix_hc;
    const size_t token_count = batch_size * seq_len;

    params.pre.resize(token_count * hc_mult);
    params.post.resize(token_count * hc_mult);
    params.comb.resize(token_count * hc_mult * hc_mult);

    std::vector<float> row(hc_mult);
    std::vector<float> comb_raw(hc_mult * hc_mult);
    for (size_t token = 0; token < token_count; ++token) {
        const float *mix = mixes.data() + token * mix_hc;
        const size_t pre_offset = token * hc_mult;

        for (size_t i = 0; i < hc_mult; ++i) {
            params.pre[pre_offset + i] = sigmoid(coeffs.scale[0] * mix[i] + coeffs.base[i]) + static_cast<float>(eps);
            params.post[pre_offset + i] =
                2.0f * sigmoid(coeffs.scale[1] * mix[hc_mult + i] + coeffs.base[hc_mult + i]);
        }

        for (size_t i = 0; i < hc_mult; ++i) {
            for (size_t j = 0; j < hc_mult; ++j) {
                const size_t idx = 2 * hc_mult + i * hc_mult + j;
                row[j] = coeffs.scale[2] * mix[idx] + coeffs.base[idx];
            }
            softmax_with_eps_inplace(row, eps);
            for (size_t j = 0; j < hc_mult; ++j) {
                comb_raw[i * hc_mult + j] = row[j];
            }
        }

        for (size_t j = 0; j < hc_mult; ++j) {
            double col_sum = eps;
            for (size_t i = 0; i < hc_mult; ++i) {
                col_sum += comb_raw[i * hc_mult + j];
            }
            for (size_t i = 0; i < hc_mult; ++i) {
                comb_raw[i * hc_mult + j] = static_cast<float>(comb_raw[i * hc_mult + j] / col_sum);
            }
        }
        for (size_t iter = 1; iter < sinkhorn_iters; ++iter) {
            for (size_t i = 0; i < hc_mult; ++i) {
                double row_sum = eps;
                for (size_t j = 0; j < hc_mult; ++j) {
                    row_sum += comb_raw[i * hc_mult + j];
                }
                for (size_t j = 0; j < hc_mult; ++j) {
                    comb_raw[i * hc_mult + j] = static_cast<float>(comb_raw[i * hc_mult + j] / row_sum);
                }
            }
            for (size_t j = 0; j < hc_mult; ++j) {
                double col_sum = eps;
                for (size_t i = 0; i < hc_mult; ++i) {
                    col_sum += comb_raw[i * hc_mult + j];
                }
                for (size_t i = 0; i < hc_mult; ++i) {
                    comb_raw[i * hc_mult + j] = static_cast<float>(comb_raw[i * hc_mult + j] / col_sum);
                }
            }
        }

        std::copy(comb_raw.begin(), comb_raw.end(), params.comb.begin() + token * hc_mult * hc_mult);
    }
}

void ensure_mhc_gpu_cache(DeepseekV4MHCCoeffs &cache,
                          const infinicore::Device &device,
                          infinicore::DataType matmul_dtype) {
    if (cache.gpu.valid && cache.gpu.device == device && cache.gpu.matmul_dtype == matmul_dtype
        && cache.gpu.mix_hc == cache.mix_hc && cache.gpu.flat_dim == cache.flat_dim) {
        return;
    }

    auto fn_tensor = float_vector_to_tensor(
        cache.fn, {cache.mix_hc, cache.flat_dim}, matmul_dtype, device);
    cache.gpu.fn_mat_right = fn_tensor->permute({1, 0})->contiguous()->view({1, cache.flat_dim, cache.mix_hc});

    std::vector<float> ones(cache.flat_dim, 1.0f);
    cache.gpu.rms_norm_weight = float_vector_to_tensor(
        ones, {cache.flat_dim}, infinicore::DataType::F32, device);

    cache.gpu.device = device;
    cache.gpu.matmul_dtype = matmul_dtype;
    cache.gpu.mix_hc = cache.mix_hc;
    cache.gpu.flat_dim = cache.flat_dim;
    cache.gpu.valid = true;
}

void ensure_mhc_head_coeffs_cached(DeepseekV4MHCCoeffs &cache,
                                  const infinicore::Tensor &base,
                                  const infinicore::Tensor &fn,
                                  const infinicore::Tensor &scale,
                                  size_t hc_mult,
                                  size_t hidden_size) {
    const size_t flat_dim = hc_mult * hidden_size;
    if (cache.mix_hc == hc_mult && cache.flat_dim == flat_dim) {
        return;
    }

    cache.base = tensor_to_float_vector(base);
    cache.fn = tensor_to_float_vector(fn);
    auto scale_values = tensor_to_float_vector(scale);
    if (cache.base.size() != hc_mult || cache.fn.size() != hc_mult * flat_dim || scale_values.empty()) {
        throw std::runtime_error("DeepseekV4MHC head: parameter shape mismatch");
    }
    cache.scale[0] = scale_values[0];
    cache.scale[1] = 1.0f;
    cache.scale[2] = 1.0f;
    cache.mix_hc = hc_mult;
    cache.flat_dim = flat_dim;
    cache.gpu.valid = false;
}

std::vector<float> compute_mhc_mixes_cpu(const infinicore::Tensor &x,
                                         const DeepseekV4MHCCoeffs &coeffs,
                                         size_t hc_mult,
                                         size_t hidden_size,
                                         double eps) {
    const auto shape = x->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    const size_t flat_dim = hc_mult * hidden_size;
    const size_t token_count = batch_size * seq_len;
    const size_t mix_hc = coeffs.mix_hc;

    auto x_values = tensor_to_float_vector(x->contiguous());
    std::vector<float> mixes(token_count * mix_hc);
    std::vector<float> flat(flat_dim);
    for (size_t token = 0; token < token_count; ++token) {
        double mean_square = 0.0;
        for (size_t i = 0; i < flat_dim; ++i) {
            flat[i] = x_values[token * flat_dim + i];
            mean_square += static_cast<double>(flat[i]) * flat[i];
        }
        const float rsqrt = static_cast<float>(1.0 / std::sqrt(mean_square / static_cast<double>(flat_dim) + eps));
        for (size_t m = 0; m < mix_hc; ++m) {
            double dot = 0.0;
            const size_t fn_offset = m * flat_dim;
            for (size_t i = 0; i < flat_dim; ++i) {
                dot += static_cast<double>(coeffs.fn[fn_offset + i]) * flat[i];
            }
            mixes[token * mix_hc + m] = static_cast<float>(dot * rsqrt);
        }
    }
    return mixes;
}

std::vector<float> compute_mhc_mixes(const infinicore::Tensor &x,
                                   const infinicore::Tensor &fn,
                                   DeepseekV4MHCCoeffs &coeffs,
                                   size_t hc_mult,
                                   size_t hidden_size,
                                   double eps) {
    const auto shape = x->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    const size_t flat_dim = hc_mult * hidden_size;

    const bool force_cpu = []() {
        if (const char *flag = std::getenv("DSV4_MHC_CPU_MIXES"); flag != nullptr && std::string(flag) == "1") {
            return true;
        }
        return false;
    }();

    if (!force_cpu && x->device().getType() != infinicore::Device::Type::CPU) {
        (void)fn;
        const size_t token_count = batch_size * seq_len;
        const size_t mix_hc = coeffs.mix_hc;

        ensure_mhc_gpu_cache(coeffs, x->device(), x->dtype());
        auto flat = x->view({batch_size, seq_len, flat_dim})->contiguous();
        flat = infinicore::op::rms_norm(
            flat, coeffs.gpu.rms_norm_weight, static_cast<float>(eps));

        // op::linear is inaccurate for large in_features; matmul dtypes must match (both bf16).
        auto flat_tokens = flat->view({token_count, 1, flat_dim});
        auto mixes = infinicore::op::matmul(flat_tokens, coeffs.gpu.fn_mat_right)
                         ->view({batch_size, seq_len, mix_hc});
        return tensor_to_float_vector(mixes->contiguous());
    }

    return compute_mhc_mixes_cpu(x, coeffs, hc_mult, hidden_size, eps);
}

} // namespace

void ensure_mhc_coeffs_cached(DeepseekV4MHCCoeffs &cache,
                              const infinicore::Tensor &base,
                              const infinicore::Tensor &fn,
                              const infinicore::Tensor &scale,
                              size_t hc_mult,
                              size_t hidden_size) {
    const size_t flat_dim = hc_mult * hidden_size;
    const size_t mix_hc = (2 + hc_mult) * hc_mult;
    if (cache.mix_hc == mix_hc && cache.flat_dim == flat_dim) {
        return;
    }

    cache.base = tensor_to_float_vector(base);
    cache.fn = tensor_to_float_vector(fn);
    auto scale_values = tensor_to_float_vector(scale);
    if (cache.base.size() != mix_hc || cache.fn.size() != mix_hc * flat_dim || scale_values.size() < 3) {
        throw std::runtime_error("DeepseekV4MHC: parameter shape mismatch");
    }
    cache.scale[0] = scale_values[0];
    cache.scale[1] = scale_values[1];
    cache.scale[2] = scale_values[2];
    cache.mix_hc = mix_hc;
    cache.flat_dim = flat_dim;
    cache.gpu.valid = false;
}

DeepseekV4MHCParams build_mhc_params(const infinicore::Tensor &x,
                                     const infinicore::Tensor &base,
                                     const infinicore::Tensor &fn,
                                     const infinicore::Tensor &scale,
                                     size_t hc_mult,
                                     size_t hidden_size,
                                     size_t sinkhorn_iters,
                                     double eps) {
    DeepseekV4MHCCoeffs coeffs;
    ensure_mhc_coeffs_cached(coeffs, base, fn, scale, hc_mult, hidden_size);

    const auto shape = x->shape();
    if (shape.size() != 4 || shape[2] != hc_mult || shape[3] != hidden_size) {
        throw std::runtime_error("DeepseekV4MHC: expected x shape [B,S,hc_mult,hidden_size]");
    }

    DeepseekV4MHCParams params;
    params.batch_size = shape[0];
    params.seq_len = shape[1];
    params.hc_mult = hc_mult;

    auto mixes = compute_mhc_mixes(x, fn, coeffs, hc_mult, hidden_size, eps);
    fill_mhc_params_from_mixes(params, mixes, coeffs, sinkhorn_iters, eps);
    return params;
}

DeepseekV4MHCPrepareResult mhc_prepare(const infinicore::Tensor &x,
                                        const infinicore::Tensor &base,
                                        const infinicore::Tensor &fn,
                                        const infinicore::Tensor &scale,
                                        DeepseekV4MHCCoeffs &coeffs_cache,
                                        size_t hc_mult,
                                        size_t hidden_size,
                                        size_t sinkhorn_iters,
                                        double eps) {
    ensure_mhc_coeffs_cached(coeffs_cache, base, fn, scale, hc_mult, hidden_size);

    const auto shape = x->shape();
    if (shape.size() != 4 || shape[2] != hc_mult || shape[3] != hidden_size) {
        throw std::runtime_error("DeepseekV4MHC: expected x shape [B,S,hc_mult,hidden_size]");
    }

    DeepseekV4MHCParams params;
    params.batch_size = shape[0];
    params.seq_len = shape[1];
    params.hc_mult = hc_mult;

    auto mixes = compute_mhc_mixes(x, fn, coeffs_cache, hc_mult, hidden_size, eps);
    fill_mhc_params_from_mixes(params, mixes, coeffs_cache, sinkhorn_iters, eps);

    DeepseekV4MHCPrepareResult result;
    result.params = std::move(params);
    result.collapsed = mhc_collapse(x, result.params);
    return result;
}

infinicore::Tensor mhc_collapse(const infinicore::Tensor &x, const DeepseekV4MHCParams &params) {
    const auto shape = x->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    const size_t hc_mult = shape[2];
    const size_t hidden_size = shape[3];

    if (x->device().getType() != infinicore::Device::Type::CPU) {
        auto pre = float_vector_to_tensor(
            params.pre, {batch_size, seq_len, hc_mult}, x->dtype(), x->device());
        const size_t token_count = batch_size * seq_len;
        auto x_tokens = x->contiguous()->view({token_count, hc_mult, hidden_size});
        auto pre_tokens = pre->view({token_count, 1, hc_mult});
        return infinicore::op::matmul(pre_tokens, x_tokens)
            ->view({batch_size, seq_len, hidden_size});
    }
    return mhc_pre(x, params);
}

infinicore::Tensor mhc_pre(const infinicore::Tensor &x,
                           const DeepseekV4MHCParams &params) {
    const auto shape = x->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    const size_t hc_mult = shape[2];
    const size_t hidden_size = shape[3];
    auto x_values = tensor_to_float_vector(x);
    std::vector<float> out(batch_size * seq_len * hidden_size, 0.0f);
    for (size_t token = 0; token < batch_size * seq_len; ++token) {
        for (size_t h = 0; h < hc_mult; ++h) {
            const float coeff = params.pre[token * hc_mult + h];
            for (size_t d = 0; d < hidden_size; ++d) {
                out[token * hidden_size + d] += coeff * x_values[(token * hc_mult + h) * hidden_size + d];
            }
        }
    }
    return float_vector_to_tensor(out, {batch_size, seq_len, hidden_size}, x->dtype(), x->device());
}

infinicore::Tensor mhc_post_gpu(const infinicore::Tensor &new_x,
                                const infinicore::Tensor &residual,
                                const DeepseekV4MHCParams &params) {
    const auto shape = residual->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    const size_t hc_mult = shape[2];
    const size_t hidden_size = shape[3];
    const size_t token_count = batch_size * seq_len;
    const auto &device = residual->device();
    const auto dtype = residual->dtype();

    auto post = float_vector_to_tensor(
        params.post, {batch_size, seq_len, hc_mult}, dtype, device);
    auto comb = float_vector_to_tensor(
        params.comb, {batch_size, seq_len, hc_mult, hc_mult}, dtype, device);

    auto res = residual->contiguous()->view({token_count, hc_mult, hidden_size});
    auto comb_out = infinicore::op::matmul(
        comb->view({token_count, hc_mult, hc_mult}), res);

    auto new_t = new_x->contiguous()->view({token_count, 1, hidden_size});
    auto post_out = infinicore::op::matmul(
        post->view({token_count, hc_mult, 1}), new_t);

    return infinicore::op::add(post_out, comb_out)
        ->view({batch_size, seq_len, hc_mult, hidden_size});
}

infinicore::Tensor mhc_post(const infinicore::Tensor &new_x,
                            const infinicore::Tensor &residual,
                            const DeepseekV4MHCParams &params) {
    const bool force_cpu = []() {
        if (const char *flag = std::getenv("DSV4_MHC_CPU_POST"); flag != nullptr && std::string(flag) == "1") {
            return true;
        }
        return false;
    }();

    if (!force_cpu && residual->device().getType() != infinicore::Device::Type::CPU) {
        return mhc_post_gpu(new_x, residual, params);
    }

    const auto shape = residual->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    const size_t hc_mult = shape[2];
    const size_t hidden_size = shape[3];
    auto residual_values = tensor_to_float_vector(residual);
    auto new_values = tensor_to_float_vector(new_x);
    std::vector<float> out(batch_size * seq_len * hc_mult * hidden_size, 0.0f);
    for (size_t token = 0; token < batch_size * seq_len; ++token) {
        for (size_t i = 0; i < hc_mult; ++i) {
            const float post = params.post[token * hc_mult + i];
            for (size_t d = 0; d < hidden_size; ++d) {
                float value = post * new_values[token * hidden_size + d];
                for (size_t j = 0; j < hc_mult; ++j) {
                    value += params.comb[(token * hc_mult + i) * hc_mult + j]
                           * residual_values[(token * hc_mult + j) * hidden_size + d];
                }
                out[(token * hc_mult + i) * hidden_size + d] = value;
            }
        }
    }
    return float_vector_to_tensor(out, residual->shape(), residual->dtype(), residual->device());
}

infinicore::Tensor expand_hc_stream(const infinicore::Tensor &hidden_states,
                                    size_t hc_mult) {
    const auto shape = hidden_states->shape();
    if (shape.size() != 3) {
        throw std::runtime_error("DeepseekV4MHC: expected hidden_states shape [B,S,D]");
    }
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    const size_t hidden_size = shape[2];

    const auto strides = hidden_states->strides();
    return hidden_states->as_strided(
                            {batch_size, seq_len, hc_mult, hidden_size},
                            {strides[0], strides[1], 0, strides[2]})
        ->contiguous();
}

infinicore::Tensor mhc_head_pre(const infinicore::Tensor &x,
                                const infinicore::Tensor &base,
                                const infinicore::Tensor &fn,
                                const infinicore::Tensor &scale,
                                DeepseekV4MHCCoeffs &coeffs_cache,
                                size_t hc_mult,
                                size_t hidden_size,
                                double eps) {
    const auto shape = x->shape();
    if (shape.size() != 4 || shape[2] != hc_mult || shape[3] != hidden_size) {
        throw std::runtime_error("DeepseekV4MHC: expected x shape [B,S,hc_mult,hidden_size]");
    }

    ensure_mhc_head_coeffs_cached(coeffs_cache, base, fn, scale, hc_mult, hidden_size);

    DeepseekV4MHCParams params;
    params.batch_size = shape[0];
    params.seq_len = shape[1];
    params.hc_mult = hc_mult;
    params.pre.resize(params.batch_size * params.seq_len * hc_mult);

    const auto mixes = compute_mhc_mixes(x, fn, coeffs_cache, hc_mult, hidden_size, eps);
    for (size_t token = 0; token < params.batch_size * params.seq_len; ++token) {
        for (size_t h = 0; h < hc_mult; ++h) {
            params.pre[token * hc_mult + h] =
                sigmoid(coeffs_cache.scale[0] * mixes[token * hc_mult + h] + coeffs_cache.base[h])
                + static_cast<float>(eps);
        }
    }

    return mhc_collapse(x, params);
}

} // namespace infinilm::models::deepseek_v4
