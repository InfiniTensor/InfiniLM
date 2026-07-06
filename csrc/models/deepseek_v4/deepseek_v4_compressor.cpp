#include "deepseek_v4_compressor.hpp"

#include "../../utils.hpp"
#include "deepseek_v4_linear.hpp"
#include "deepseek_v4_utils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace infinilm::models::deepseek_v4 {
namespace {

float stable_exp(float value) { return std::exp(value); }

float round_to_dtype(float value, infinicore::DataType dtype) {
    switch (dtype) {
    case infinicore::DataType::F32:
        return value;
    case infinicore::DataType::F16:
        return f16_to_f32(f32_to_f16(value));
    case infinicore::DataType::BF16:
        return bf16_to_f32(f32_to_bf16(value));
    default:
        return value;
    }
}

void apply_rms_norm_inplace(std::vector<float> &values,
                            const std::vector<float> &weight,
                            size_t hidden_size,
                            double eps,
                            infinicore::DataType dtype) {
    if (hidden_size == 0 || weight.size() != hidden_size) {
        throw std::runtime_error("DeepseekV4Compressor: RMSNorm weight shape mismatch");
    }
    const size_t groups = values.size() / hidden_size;
    for (size_t group = 0; group < groups; ++group) {
        const size_t offset = group * hidden_size;
        double mean_square = 0.0;
        for (size_t d = 0; d < hidden_size; ++d) {
            values[offset + d] = round_to_dtype(values[offset + d], dtype);
            const float value = values[offset + d];
            mean_square += static_cast<double>(value) * value;
        }
        const float rsqrt = static_cast<float>(
            1.0 / std::sqrt(mean_square / static_cast<double>(hidden_size) + eps));
        for (size_t d = 0; d < hidden_size; ++d) {
            values[offset + d] = round_to_dtype(values[offset + d] * rsqrt * weight[d], dtype);
        }
    }
}

} // namespace

DeepseekV4Compressor::DeepseekV4Compressor(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                           size_t compress_ratio,
                                           size_t head_dim,
                                           const infinicore::Device &device)
    : compress_ratio_(compress_ratio),
      head_dim_(head_dim),
      coff_(compress_ratio == 4 ? 2 : 1),
      rms_norm_eps_(model_config->get<double>("rms_norm_eps")) {
    const auto &dtype = model_config->get_dtype();
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t compressed_dim = coff_ * head_dim;

    auto none_quantization = deepseek_v4_linear_quantization(model_config, false);
    INFINICORE_NN_PARAMETER_INIT(ape, ({compress_ratio, compressed_dim}, dtype, device));
    INFINICORE_NN_MODULE_INIT(wkv, hidden_size, compressed_dim, none_quantization, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(wgate, hidden_size, compressed_dim, none_quantization, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(norm, head_dim, rms_norm_eps_, dtype, device);
}

void DeepseekV4Compressor::process_weights_after_loading() {
    if (ape_converted_ || coff_ != 2) {
        return;
    }

    auto ape = tensor_to_float_vector(ape_);
    const size_t compressed_dim = coff_ * head_dim_;
    if (ape.size() != compress_ratio_ * compressed_dim) {
        throw std::runtime_error("DeepseekV4Compressor: unexpected APE shape");
    }

    std::vector<float> converted(ape.size());
    for (size_t row = 0; row < compress_ratio_; ++row) {
        for (size_t col = 0; col < compressed_dim; ++col) {
            const size_t flat = row * compressed_dim + col;
            const size_t cat_row = flat / head_dim_;
            const size_t cat_col = flat % head_dim_;
            const bool second_half = cat_row >= compress_ratio_;
            const size_t src_row = second_half ? cat_row - compress_ratio_ : cat_row;
            const size_t src_col = (second_half ? head_dim_ : 0) + cat_col;
            converted[flat] = ape[src_row * compressed_dim + src_col];
        }
    }

    auto converted_tensor = float_vector_to_tensor(converted, ape_->shape(), ape_->dtype(), ape_->device());
    ape_->copy_from(converted_tensor);
    ape_host_ = tensor_to_float_vector(ape_);
    ape_host_cached_ = true;
    ape_converted_ = true;
}

void DeepseekV4Compressor::ensure_host_caches() const {
    if (!ape_host_cached_) {
        ape_host_ = tensor_to_float_vector(ape_);
        ape_host_cached_ = true;
    }
    if (!norm_weight_host_cached_) {
        norm_weight_host_ = tensor_to_float_vector(norm_->weight());
        norm_weight_host_cached_ = true;
    }
}

std::vector<float> DeepseekV4Compressor::forward_values(const infinicore::Tensor &hidden_states,
                                                        size_t &batch_size,
                                                        size_t &num_blocks) const {
    const auto shape = hidden_states->shape();
    batch_size = shape[0];
    const size_t seq_len = shape[1];
    auto hidden_states_mut = hidden_states;
    auto kv_t = wkv_->forward(hidden_states_mut);
    auto score_t = wgate_->forward(hidden_states_mut);
    auto kv = tensor_to_float_vector(kv_t);
    auto score = tensor_to_float_vector(score_t);
    ensure_host_caches();
    const auto &ape = ape_host_;
    const size_t m = compress_ratio_;
    const size_t compressed_dim = coff_ * head_dim_;
    const size_t usable_len = (seq_len / m) * m;
    num_blocks = usable_len / m;
    std::vector<float> out(batch_size * num_blocks * head_dim_, 0.0f);

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t block = 0; block < num_blocks; ++block) {
            const size_t pool_len = coff_ == 2 ? 2 * m : m;
            for (size_t d = 0; d < head_dim_; ++d) {
                std::vector<float> values(pool_len, 0.0f);
                std::vector<float> scores(pool_len, -std::numeric_limits<float>::infinity());
                for (size_t r = 0; r < m; ++r) {
                    const size_t src = block * m + r;
                    if (src < seq_len) {
                        const size_t src_offset = (b * seq_len + src) * compressed_dim;
                        if (coff_ == 1) {
                            values[r] = kv[src_offset + d];
                            scores[r] = score[src_offset + d] + ape[r * compressed_dim + d];
                        } else {
                            values[m + r] = kv[src_offset + head_dim_ + d];
                            scores[m + r] = score[src_offset + head_dim_ + d] + ape[r * compressed_dim + head_dim_ + d];
                        }
                    }
                    if (coff_ == 2 && block > 0) {
                        const size_t prev = (block - 1) * m + r;
                        if (prev < seq_len) {
                            const size_t prev_offset = (b * seq_len + prev) * compressed_dim;
                            values[r] = kv[prev_offset + d];
                            scores[r] = score[prev_offset + d] + ape[r * compressed_dim + d];
                        }
                    }
                }
                float max_score = -std::numeric_limits<float>::infinity();
                for (float v : scores) {
                    max_score = std::max(max_score, v);
                }
                double denom = 0.0;
                for (float v : scores) {
                    if (std::isfinite(v)) {
                        denom += stable_exp(v - max_score);
                    }
                }
                float value = 0.0f;
                if (denom > 0.0) {
                    for (size_t i = 0; i < pool_len; ++i) {
                        if (std::isfinite(scores[i])) {
                            value += static_cast<float>(stable_exp(scores[i] - max_score) / denom) * values[i];
                        }
                    }
                }
                out[(b * num_blocks + block) * head_dim_ + d] = value;
            }
        }
    }
    apply_rms_norm_inplace(out, norm_weight_host_, head_dim_, rms_norm_eps_, hidden_states->dtype());
    return out;
}

infinicore::Tensor DeepseekV4Compressor::forward(const infinicore::Tensor &hidden_states) const {
    size_t batch_size = 0;
    size_t num_blocks = 0;
    auto out = forward_values(hidden_states, batch_size, num_blocks);
    return float_vector_to_tensor(out, {batch_size, num_blocks, head_dim_},
                                  hidden_states->dtype(), hidden_states->device());
}

} // namespace infinilm::models::deepseek_v4
