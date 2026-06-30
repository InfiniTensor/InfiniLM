#include "deepseek_v4_compressor.hpp"

#include "deepseek_v4_linear.hpp"
#include "deepseek_v4_utils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace infinilm::models::deepseek_v4 {
namespace {

float stable_exp(float value) { return std::exp(value); }

} // namespace

DeepseekV4Compressor::DeepseekV4Compressor(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                           size_t compress_ratio,
                                           size_t head_dim,
                                           const infinicore::Device &device)
    : compress_ratio_(compress_ratio),
      head_dim_(head_dim),
      coff_(compress_ratio == 4 ? 2 : 1) {
    const auto &dtype = model_config->get_dtype();
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const double rms_norm_eps = model_config->get<double>("rms_norm_eps");
    const size_t compressed_dim = coff_ * head_dim;

    auto none_quantization = deepseek_v4_linear_quantization(model_config, false);
    INFINICORE_NN_PARAMETER_INIT(ape, ({compress_ratio, compressed_dim}, dtype, device));
    INFINICORE_NN_MODULE_INIT(wkv, hidden_size, compressed_dim, none_quantization, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(wgate, hidden_size, compressed_dim, none_quantization, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(norm, head_dim, rms_norm_eps, dtype, device);
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
    auto ape = tensor_to_float_vector(ape_);
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
    auto pooled = float_vector_to_tensor(out, {batch_size, num_blocks, head_dim_},
                                         hidden_states->dtype(), hidden_states->device());
    return tensor_to_float_vector(norm_->forward(pooled));
}

infinicore::Tensor DeepseekV4Compressor::forward(const infinicore::Tensor &hidden_states) const {
    size_t batch_size = 0;
    size_t num_blocks = 0;
    auto out = forward_values(hidden_states, batch_size, num_blocks);
    return float_vector_to_tensor(out, {batch_size, num_blocks, head_dim_},
                                  hidden_states->dtype(), hidden_states->device());
}

} // namespace infinilm::models::deepseek_v4
