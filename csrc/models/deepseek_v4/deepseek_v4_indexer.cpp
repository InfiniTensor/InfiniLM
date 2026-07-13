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

void DeepseekV4Indexer::reset_runtime_state() const {
    compressed_cache_.reset();
    compressed_cache_storage_.reset();
    compressed_cache_batch_ = 0;
    compressed_cache_blocks_ = 0;
    compressed_cache_capacity_ = 0;
}

void DeepseekV4Indexer::set_compressed_cache(
    const infinicore::Tensor &compressed,
    size_t batch_size,
    size_t num_blocks) const {
    compressed_cache_ = compressed;
    compressed_cache_storage_.reset();
    compressed_cache_batch_ = batch_size;
    compressed_cache_blocks_ = num_blocks;
    compressed_cache_capacity_ = num_blocks;
}

void DeepseekV4Indexer::append_compressed_cache(
    const infinicore::Tensor &new_blocks) const {
    const auto shape = new_blocks->shape();
    const size_t batch_size = shape[0];
    const size_t append_blocks = shape[1];
    if (append_blocks == 0) {
        return;
    }

    const size_t old_blocks = compressed_cache_blocks_;
    const size_t new_block_count = old_blocks + append_blocks;
    if (!compressed_cache_storage_
        || compressed_cache_capacity_ < new_block_count
        || compressed_cache_batch_ != batch_size) {
        size_t new_capacity = std::max(new_block_count, old_blocks + static_cast<size_t>(16));
        new_capacity = std::max(new_capacity, compressed_cache_capacity_ * 2);
        std::vector<size_t> storage_shape{batch_size, new_capacity, shape[2]};
        auto next_storage = infinicore::Tensor::empty(
            storage_shape, new_blocks->dtype(), new_blocks->device());
        if (compressed_cache_ && old_blocks > 0
            && compressed_cache_batch_ == batch_size) {
            next_storage->narrow({{1, 0, old_blocks}})->copy_from(compressed_cache_);
        }
        compressed_cache_storage_ = next_storage;
        compressed_cache_capacity_ = new_capacity;
    }

    compressed_cache_storage_->narrow({{1, old_blocks, append_blocks}})
        ->copy_from(new_blocks);
    compressed_cache_ = compressed_cache_storage_->narrow(
        {{1, 0, new_block_count}});
    compressed_cache_batch_ = batch_size;
    compressed_cache_blocks_ = new_block_count;
}

infinicore::Tensor DeepseekV4Indexer::get_or_update_compressed_cache(
    const infinicore::Tensor &hidden_states,
    size_t batch_size,
    size_t expected_blocks,
    size_t query_len) const {
    if (compressed_cache_
        && compressed_cache_batch_ == batch_size
        && compressed_cache_blocks_ == expected_blocks) {
        return compressed_cache_;
    }

    if (query_len == 1
        && compressed_cache_
        && compressed_cache_batch_ == batch_size
        && expected_blocks == compressed_cache_blocks_ + 1
        && hidden_states->shape()[1] >= 2 * compress_ratio_) {
        const size_t recent_len = 2 * compress_ratio_;
        const size_t total_len = hidden_states->shape()[1];
        auto recent_hidden = hidden_states->narrow(
            {{1, total_len - recent_len, recent_len}})->contiguous();
        size_t recent_batch = 0;
        size_t recent_blocks = 0;
        auto recent_compressed = compressor_->forward_tensor(
            recent_hidden, recent_batch, recent_blocks);
        if (recent_batch == batch_size && recent_blocks > 0) {
            auto newest_block = recent_compressed->narrow(
                {{1, recent_blocks - 1, 1}})->contiguous();
            append_compressed_cache(newest_block);
            if (compressed_cache_blocks_ == expected_blocks) {
                return compressed_cache_;
            }
        }
    }

    size_t compressed_batch = 0;
    size_t compressed_blocks = 0;
    auto compressed = compressor_->forward_tensor(
        hidden_states, compressed_batch, compressed_blocks);
    if (compressed_batch != batch_size || compressed_blocks != expected_blocks) {
        throw std::runtime_error("DeepseekV4Indexer: compressed cache shape mismatch");
    }
    set_compressed_cache(compressed, compressed_batch, compressed_blocks);
    return compressed_cache_;
}

infinicore::Tensor DeepseekV4Indexer::forward_tensor(const infinicore::Tensor &hidden_states,
                                                     const infinicore::Tensor &q_residual,
                                                     const std::vector<int64_t> &positions,
                                                     size_t &top_k,
                                                     size_t query_start,
                                                     size_t query_len,
                                                     size_t logical_total_len) const {
    if (hidden_states->device().getType() == infinicore::Device::Type::CPU) {
        throw std::runtime_error("DeepseekV4Indexer: GPU tensor required");
    }

    const auto shape = hidden_states->shape();
    const size_t batch_size = shape[0];
    const size_t available_hidden_len = shape[1];
    const size_t total_len = logical_total_len == 0 ? available_hidden_len : logical_total_len;
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

    const size_t num_blocks = total_len / compress_ratio_;
    top_k = std::min(index_topk_, num_blocks);
    if (num_blocks == 0 || top_k == 0 || query_len == 0) {
        top_k = 0;
        return infinicore::Tensor();
    }
    auto compressed = get_or_update_compressed_cache(
        hidden_states, batch_size, num_blocks, query_len);
    if (use_indexer_full_coverage_shortcut() && top_k == num_blocks) {
        top_k = 0;
        return infinicore::Tensor();
    }

    top_k = std::min(index_topk_, num_blocks);
    if (num_blocks == 0 || top_k == 0) {
        top_k = 0;
        return infinicore::Tensor();
    }

    auto q_input = q_residual;
    auto weights_input = hidden_states;
    if (query_start != 0 || query_len != total_len
        || q_residual->shape()[1] != total_len) {
        const size_t q_available = q_residual->shape()[1];
        if (q_available < query_len) {
            throw std::runtime_error("DeepseekV4Indexer: recent query history is too short");
        }
        const size_t q_start = q_available == total_len
                                 ? query_start
                                 : q_available - query_len;
        q_input = q_residual->narrow({{1, q_start, query_len}})->contiguous();
    }
    if (query_start != 0 || query_len != total_len
        || available_hidden_len != total_len) {
        if (available_hidden_len < query_len) {
            throw std::runtime_error("DeepseekV4Indexer: recent hidden history is too short");
        }
        weights_input = hidden_states->narrow(
            {{1, available_hidden_len - query_len, query_len}})->contiguous();
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
