#include "deepseek_v4_utils.hpp"

#include "../../utils.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/ops/add.hpp"
#include "infinicore/ops/deepseek_v4_mhc.hpp"
#include "infinicore/ops/deepseek_v4_mhc_head.hpp"
#include "infinicore/ops/deepseek_v4_mhc_post.hpp"
#include "infinicore/ops/matmul.hpp"
#include "infinicore/ops/unweighted_rms_norm.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
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

std::tuple<infinicore::Tensor, infinicore::Tensor, infinicore::Tensor>
mhc_prepare(const infinicore::Tensor &x,
            const infinicore::Tensor &base,
            const infinicore::Tensor &fn_mat_right,
            const infinicore::Tensor &scale,
            size_t hc_mult,
            size_t hidden_size,
            size_t sinkhorn_iters,
            double rms_norm_eps,
            double hc_eps) {
    const auto shape = x->shape();
    if (shape.size() != 4 || shape[2] != hc_mult || shape[3] != hidden_size) {
        throw std::runtime_error("DeepseekV4MHC: expected x shape [B,S,hc_mult,hidden_size]");
    }
    if (x->device().getType() == infinicore::Device::Type::CPU) {
        throw std::runtime_error("DeepseekV4MHC: mhc_prepare requires GPU tensor");
    }

    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    const size_t flat_dim = hc_mult * hidden_size;
    const size_t token_count = batch_size * seq_len;
    const size_t mix_hc = fn_mat_right->shape()[2];

    auto x_view = x->is_contiguous() ? x : x->contiguous();
    auto flat = x_view->view({batch_size, seq_len, flat_dim});
    infinicore::Tensor mixes;
    static const bool use_raw_mhc_gemm = [] {
        const char *raw = std::getenv("INFINILM_DSV4_RAW_MHC_GEMM");
        return raw == nullptr || std::string(raw) != "0";
    }();
    if (use_raw_mhc_gemm) {
        auto raw_mixes = infinicore::op::matmul(flat->view({token_count, 1, flat_dim}), fn_mat_right)
                             ->view({batch_size, seq_len, mix_hc});
        mixes = infinicore::op::deepseek_v4_mhc_scale_mixes(
            x_view, raw_mixes, static_cast<float>(rms_norm_eps));
    } else {
        auto normed_flat = infinicore::op::unweighted_rms_norm(flat, static_cast<float>(rms_norm_eps));
        mixes = infinicore::op::matmul(normed_flat->view({token_count, 1, flat_dim}), fn_mat_right)
                     ->view({batch_size, seq_len, mix_hc});
    }

    static const bool use_fused_mhc_pre = [] {
        const char *raw = std::getenv("INFINILM_DSV4_FUSED_MHC_PRE");
        return raw == nullptr || std::string(raw) != "0";
    }();
    if (use_fused_mhc_pre) {
        auto [collapsed, post, comb] = infinicore::op::deepseek_v4_mhc_pre_collapse(
            x_view, mixes, base, scale, sinkhorn_iters, static_cast<float>(hc_eps));
        return {collapsed, post, comb};
    }

    auto [pre, post, comb] = infinicore::op::deepseek_v4_mhc_params(
        mixes, base, scale, sinkhorn_iters, static_cast<float>(hc_eps));
    auto x_tokens = x_view->view({token_count, hc_mult, hidden_size});
    auto pre_tokens = pre->view({token_count, 1, hc_mult});
    auto collapsed = infinicore::op::matmul(pre_tokens, x_tokens)
                         ->view({batch_size, seq_len, hidden_size});
    return {collapsed, post, comb};
}

infinicore::Tensor mhc_post_gpu(const infinicore::Tensor &new_x,
                                const infinicore::Tensor &residual,
                                const infinicore::Tensor &post,
                                const infinicore::Tensor &comb) {
    auto residual_view = residual->is_contiguous() ? residual : residual->contiguous();
    auto new_x_view = new_x->is_contiguous() ? new_x : new_x->contiguous();
    auto post_view = post->is_contiguous() ? post : post->contiguous();
    auto comb_view = comb->is_contiguous() ? comb : comb->contiguous();
    return infinicore::op::deepseek_v4_mhc_post(new_x_view, residual_view, post_view, comb_view);
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
                                const infinicore::Tensor &fn_mat_right,
                                const infinicore::Tensor &scale,
                                size_t hc_mult,
                                size_t hidden_size,
                                double eps) {
    const auto shape = x->shape();
    if (shape.size() != 4 || shape[2] != hc_mult || shape[3] != hidden_size) {
        throw std::runtime_error("DeepseekV4MHC: expected x shape [B,S,hc_mult,hidden_size]");
    }

    if (x->device().getType() == infinicore::Device::Type::CPU) {
        throw std::runtime_error("DeepseekV4MHC head requires GPU tensor");
    }

    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    const size_t flat_dim = hc_mult * hidden_size;
    const size_t token_count = batch_size * seq_len;
    const size_t mix_hc = fn_mat_right->shape()[2];

    auto x_view = x->is_contiguous() ? x : x->contiguous();
    auto flat = x_view->view({batch_size, seq_len, flat_dim});
    flat = infinicore::op::unweighted_rms_norm(flat, static_cast<float>(eps));
    auto mixes = infinicore::op::matmul(flat->view({token_count, 1, flat_dim}),
                                        fn_mat_right)
                     ->view({batch_size, seq_len, mix_hc});
    return infinicore::op::deepseek_v4_mhc_head_collapse(
        x_view, mixes, base, scale, static_cast<float>(eps));
}

} // namespace infinilm::models::deepseek_v4
