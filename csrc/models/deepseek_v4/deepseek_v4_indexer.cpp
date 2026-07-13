#include "deepseek_v4_indexer.hpp"

#include "deepseek_v4_linear.hpp"
#include "deepseek_v4_utils.hpp"
#include "infinicore/ops/deepseek_v4_indexer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace infinilm::models::deepseek_v4 {
namespace {

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
    INFINICORE_NN_MODULE_INIT(wq_b, q_lora_rank, index_head_dim_ * index_n_heads_,
                              quantization_method, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(weights_proj, hidden_size_, index_n_heads_,
                              none_quantization, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(compressor, model_config, compress_ratio_, index_head_dim_, device);
}

infinicore::Tensor DeepseekV4Indexer::forward_tensor(const infinicore::Tensor &hidden_states,
                                                     const infinicore::Tensor &q_residual,
                                                     const std::vector<int64_t> &positions,
                                                     size_t &top_k,
                                                     size_t query_start,
                                                     size_t query_len) const {
    if (hidden_states->device().getType() == infinicore::Device::Type::CPU) {
        throw std::runtime_error("DeepseekV4Indexer: GPU tensor required");
    }

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
    auto compressed = compressor_->forward_tensor(hidden_states, comp_batch, num_blocks);
    if (comp_batch != batch_size) {
        throw std::runtime_error("DeepseekV4Indexer: compressed batch mismatch");
    }
    top_k = std::min(index_topk_, num_blocks);
    if (num_blocks == 0 || top_k == 0) {
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
    auto weights = weights_proj_->forward(weights_input);
    auto positions_tensor = int64_vector_to_tensor(
        positions, {positions.size()}, q_proj->device());
    return infinicore::op::deepseek_v4_indexer(
        q_proj->contiguous(),
        weights->contiguous(),
        compressed->contiguous(),
        positions_tensor,
        top_k,
        query_start,
        compress_ratio_);
}

} // namespace infinilm::models::deepseek_v4
