#pragma once

#include "../cache/cache.hpp"
#include "../config/model_config.hpp"
#include "../engine/distributed/distributed.hpp"
#include "infinicore/nn/module.hpp"
#include "nlohmann/json.hpp"

#include <any>
#include <memory>

#include <optional>

namespace infinilm {
class InfinilmModel : public infinicore::nn::Module {
public:
    struct Config {
        std::string model_type;
        virtual ~Config() = default;
    };

    struct Input {
        /// Token IDs tensor of shape `[batch, seq_len]`.
        std::optional<infinicore::Tensor> input_ids;
        /// Position IDs tensor of shape `[batch, seq_len]` or `[seq_len]`.
        std::optional<infinicore::Tensor> position_ids;
        /// Past Lengths of cached sequence for each request, of shape `[num_requests]`.
        std::optional<infinicore::Tensor> past_sequence_lengths;
        /// ToTal Lengths for each request sequence, of shape `[num_requests]`.
        std::optional<infinicore::Tensor> total_sequence_lengths;
        /// Offsets of each request in a continous-batched sequence, of shape `[num_requests + 1]`.
        std::optional<infinicore::Tensor> input_offsets;
        /// Cumulative total sequence lengths for each request, of shape `[num_requests + 1]`.
        std::optional<infinicore::Tensor> cu_seqlens;
        /// Block ids for each request `[batch, max_block_table_length]`. Used for paged cache.
        std::optional<infinicore::Tensor> block_tables;
        /// Slot ids for each token `[seq]`. Used for paged cache.
        std::optional<infinicore::Tensor> slot_mapping;
    };

    struct Output {
        /// Logits.
        infinicore::Tensor logits;
    };

    std::shared_ptr<cache::Cache> kv_cache_;

protected:
    std::shared_ptr<infinilm::config::ModelConfig> model_config_;
    engine::distributed::RankInfo rank_info_;
    std::unique_ptr<cache::CacheConfig> cache_config_;

public:
    virtual ~InfinilmModel() = default;
    virtual Output forward(const Input &input) const
        = 0;

    void reset_cache(const cache::CacheConfig *cache_config) {
        if (cache_config == nullptr) {
            kv_cache_ = nullptr;
            cache_config_ = nullptr;
            return;
        }

        cache_config_ = cache_config->unique_copy();

        if (model_config_ == nullptr) {
            throw std::runtime_error("model_config_ is not initialized");
        }

        if (auto kv_cache_config = dynamic_cast<const cache::StaticKVCacheConfig *>(cache_config_.get())) {
            kv_cache_ = std::make_shared<cache::StaticKVCache>(
                model_config_->get_head_dim(),
                model_config_->get_head_dim(),
                model_config_->get<size_t>("num_key_value_heads"),
                model_config_->get<size_t>("num_key_value_heads"),
                model_config_->get<size_t>("num_hidden_layers"),
                model_config_->get<size_t>("max_position_embeddings"),
                model_config_->get_dtype(),
                *kv_cache_config,
                rank_info_);
        } else if (auto paged_kv_cache_config = dynamic_cast<const cache::PagedKVCacheConfig *>(cache_config_.get())) {
            kv_cache_ = std::make_shared<cache::PagedKVCache>(
                model_config_->get_head_dim(),
                model_config_->get_head_dim(),
                model_config_->get<size_t>("num_key_value_heads"),
                model_config_->get<size_t>("num_key_value_heads"),
                model_config_->get<size_t>("num_hidden_layers"),
                model_config_->get_dtype(),
                *paged_kv_cache_config,
                rank_info_);
        } else {
            throw std::runtime_error("Unsupported cache type");
        }
    }

    const cache::CacheConfig *get_cache_config() const {
        return cache_config_.get();
    }
};
} // namespace infinilm
