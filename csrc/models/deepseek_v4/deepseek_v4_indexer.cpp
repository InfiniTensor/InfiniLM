#include "deepseek_v4_indexer.hpp"

#include "deepseek_v4_linear.hpp"
#include "deepseek_v4_utils.hpp"
#include "infinicore/ops/deepseek_v4_indexer.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace infinilm::models::deepseek_v4 {
namespace {

bool force_indexer_cpu() {
    static const bool value = [] {
        if (const char *flag = std::getenv("DSV4_INDEXER_CPU"); flag != nullptr) {
            return std::string(flag) == "1";
        }
        return false;
    }();
    return value;
}

bool use_indexer_fused_gpu() {
    static const bool value = [] {
        if (const char *flag = std::getenv("DSV4_INDEXER_FUSED_GPU"); flag != nullptr) {
            return std::string(flag) != "0";
        }
        return true;
    }();
    return value;
}

bool use_indexer_full_coverage_shortcut() {
    static const bool value = [] {
        if (const char *flag = std::getenv("DSV4_INDEXER_FULL_COVERAGE_SHORTCUT"); flag != nullptr) {
            return std::string(flag) != "0";
        }
        return true;
    }();
    return value;
}

} // namespace

DeepseekV4Indexer::DeepseekV4Indexer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                     size_t compress_ratio,
                                     const infinicore::Device &device)
    : hidden_size_(model_config->get<size_t>("hidden_size")),
      compress_ratio_(compress_ratio),
      index_head_dim_(model_config->get<size_t>("index_head_dim")),
      index_n_heads_(model_config->get<size_t>("index_n_heads")),
      index_topk_(model_config->get<size_t>("index_topk")) {
    const auto &dtype = model_config->get_dtype();
    const size_t q_lora_rank = model_config->get<size_t>("q_lora_rank");

    auto quantization_method = deepseek_v4_linear_quantization(model_config, true);
    auto none_quantization = deepseek_v4_linear_quantization(model_config, false);
    INFINICORE_NN_MODULE_INIT(wq_b, q_lora_rank, index_head_dim_ * index_n_heads_, quantization_method, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(weights_proj, hidden_size_, index_n_heads_, none_quantization, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(compressor, model_config, compress_ratio_, index_head_dim_, device);
}

infinicore::Tensor DeepseekV4Indexer::forward_tensor(const infinicore::Tensor &hidden_states,
                                                     const infinicore::Tensor &q_residual,
                                                     const std::vector<int64_t> &positions,
                                                     size_t &top_k,
                                                     size_t query_start,
                                                     size_t query_len) const {
    const auto shape = hidden_states->shape();
    const size_t batch_size = shape[0];
    const size_t total_len = shape[1];
    if (positions.size() < total_len) {
        throw std::runtime_error("DeepseekV4Indexer: positions length mismatch");
    }
    if (query_start > total_len) {
        throw std::runtime_error("DeepseekV4Indexer: query_start out of range");
    }
    if (query_len == 0) {
        query_len = total_len - query_start;
    }
    if (query_start + query_len > total_len) {
        throw std::runtime_error("DeepseekV4Indexer: query range out of bounds");
    }

    if (compress_ratio_ == 0) {
        throw std::runtime_error("DeepseekV4Indexer: compress_ratio must be non-zero");
    }
    const size_t cheap_num_blocks = total_len / compress_ratio_;
    top_k = std::min(index_topk_, cheap_num_blocks);
    if (cheap_num_blocks == 0 || top_k == 0 || query_len == 0) {
        top_k = 0;
        return infinicore::Tensor();
    }
    if (use_indexer_full_coverage_shortcut() && top_k == cheap_num_blocks) {
        top_k = 0;
        return infinicore::Tensor();
    }

    size_t comp_batch = 0;
    size_t num_blocks = 0;
    infinicore::Tensor compressed_gpu;
    std::vector<float> compressed;
    const bool want_fused_gpu = !force_indexer_cpu() && use_indexer_fused_gpu()
                             && hidden_states->device().getType() != infinicore::Device::Type::CPU;
    if (want_fused_gpu) {
        try {
            compressed_gpu = compressor_->forward_tensor(hidden_states, comp_batch, num_blocks);
        } catch (const std::exception &) {
            compressed_gpu = infinicore::Tensor();
        }
    }
    if (!compressed_gpu) {
        compressed = compressor_->forward_values(hidden_states, comp_batch, num_blocks);
    }
    if (comp_batch != batch_size) {
        throw std::runtime_error("DeepseekV4Indexer: compressed batch mismatch");
    }
    top_k = std::min(index_topk_, num_blocks);
    if (num_blocks == 0 || top_k == 0 || query_len == 0) {
        top_k = 0;
        return infinicore::Tensor();
    }

    auto q_input = q_residual;
    auto weights_input = hidden_states;
    if (query_start != 0 || query_len != total_len) {
        q_input = q_residual->narrow({{1, query_start, query_len}})->contiguous();
        weights_input = hidden_states->narrow({{1, query_start, query_len}})->contiguous();
    }

    auto q_proj = wq_b_->forward(q_input)
                      ->view({batch_size, query_len, index_n_heads_, index_head_dim_});
    auto weights_proj = weights_proj_->forward(weights_input);
    const float score_scale = 1.0f / std::sqrt(static_cast<float>(index_head_dim_));
    const float weight_scale = 1.0f / std::sqrt(static_cast<float>(index_n_heads_));

    if (!force_indexer_cpu() && use_indexer_fused_gpu() && hidden_states->device().getType() != infinicore::Device::Type::CPU) {
        try {
            auto positions_tensor = int64_vector_to_tensor(positions, {positions.size()}, q_proj->device());
            return infinicore::op::deepseek_v4_indexer(
                q_proj->contiguous(),
                weights_proj->contiguous(),
                compressed_gpu ? compressed_gpu->contiguous()
                               : float_vector_to_tensor(compressed,
                                                        {batch_size, num_blocks, index_head_dim_},
                                                        q_proj->dtype(),
                                                        q_proj->device()),
                positions_tensor,
                top_k,
                query_start,
                compress_ratio_);
        } catch (const std::exception &) {
            // Fall back to the reference CPU path if the fused op is unavailable.
        }
    }

    if (compressed.empty()) {
        size_t host_batch = 0;
        size_t host_num_blocks = 0;
        compressed = compressor_->forward_values(hidden_states, host_batch, host_num_blocks);
        if (host_batch != batch_size || host_num_blocks != num_blocks) {
            throw std::runtime_error("DeepseekV4Indexer: host compressed shape mismatch");
        }
    }

    auto q_values = tensor_to_float_vector(q_proj);
    auto weights = tensor_to_float_vector(weights_proj);
    std::vector<int64_t> indices(batch_size * query_len * top_k, -1);

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t tq = 0; tq < query_len; ++tq) {
            const size_t t = query_start + tq;
            const int64_t causal_threshold = (positions[t] + 1) / static_cast<int64_t>(compress_ratio_);
            std::vector<std::pair<float, int64_t>> ranked;
            ranked.reserve(num_blocks);
            for (size_t block = 0; block < num_blocks; ++block) {
                float score_sum = 0.0f;
                if (static_cast<int64_t>(block) < causal_threshold) {
                    for (size_t h = 0; h < index_n_heads_; ++h) {
                        const size_t q_offset = ((b * query_len + tq) * index_n_heads_ + h) * index_head_dim_;
                        const size_t k_offset = (b * num_blocks + block) * index_head_dim_;
                        double dot = 0.0;
                        for (size_t d = 0; d < index_head_dim_; ++d) {
                            dot += static_cast<double>(q_values[q_offset + d]) * compressed[k_offset + d];
                        }
                        const float relu_score = std::max(0.0f, static_cast<float>(dot) * score_scale);
                        score_sum += relu_score * weights[(b * query_len + tq) * index_n_heads_ + h] * weight_scale;
                    }
                } else {
                    score_sum = -std::numeric_limits<float>::infinity();
                }
                ranked.emplace_back(score_sum, static_cast<int64_t>(block));
            }
            const size_t row_top_k = std::min(top_k, ranked.size());
            std::partial_sort(ranked.begin(), ranked.begin() + static_cast<std::ptrdiff_t>(row_top_k), ranked.end(),
                              [](const auto &a, const auto &b) {
                                  if (a.first == b.first) {
                                      return a.second < b.second;
                                  }
                                  return a.first > b.first;
                              });
            for (size_t k = 0; k < row_top_k; ++k) {
                const bool valid = std::isfinite(ranked[k].first)
                                && ranked[k].second < causal_threshold;
                indices[(b * query_len + tq) * top_k + k] = valid ? ranked[k].second : -1;
            }
        }
    }
    return int64_vector_to_tensor(indices, {indices.size()}, hidden_states->device());
}

std::vector<int64_t> DeepseekV4Indexer::forward(const infinicore::Tensor &hidden_states,
                                                const infinicore::Tensor &q_residual,
                                                const std::vector<int64_t> &positions,
                                                size_t &top_k,
                                                size_t query_start,
                                                size_t query_len) const {
    auto indices = forward_tensor(hidden_states, q_residual, positions, top_k, query_start, query_len);
    if (!indices) {
        return {};
    }
    return tensor_to_int64_vector(indices);
}

} // namespace infinilm::models::deepseek_v4
