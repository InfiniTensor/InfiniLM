#pragma once

#include "../backends/attention_backends.hpp"
#include "../cache/cache.hpp"
#include "../config/infinilm_config.hpp"
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

    backends::AttentionBackend attention_backend_;

protected:
    /*
    https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu_model_runner.py

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        kv_cache_config = deepcopy(kv_cache_config)
        self.kv_cache_config = kv_cache_config
    */
    void initialize_kv_cache(const cache::CacheConfig *cache_config,
                             const std::shared_ptr<infinilm::config::ModelConfig> text_model_config);

    void reset_text_cache(const cache::CacheConfig *cache_config,
                          const std::shared_ptr<infinilm::config::ModelConfig> text_config) {
        if (cache_config == nullptr) {
            kv_cache_ = nullptr;
            cache_config_ = nullptr;
            return;
        }
        if (text_config == nullptr) {
            throw std::runtime_error("txt_model_config is not initialized");
        }
        cache_config_ = cache_config->unique_copy();

        const backends::AttentionBackend attention_backend = config::get_current_infinilm_config().attention_backend;

        switch (attention_backend) {
        case backends::AttentionBackend::STATIC_ATTN: {
            auto static_kv_cache_config = dynamic_cast<const cache::StaticKVCacheConfig *>(cache_config_.get());
            if (nullptr == static_kv_cache_config) {
                throw std::runtime_error("static_kv_cache_config is not initialized");
            }
            kv_cache_ = std::make_shared<cache::StaticKVCache>(
                text_config->get_head_dim(),
                text_config->get_head_dim(),
                text_config->get<size_t>("num_key_value_heads"),
                text_config->get<size_t>("num_key_value_heads"),
                text_config->get<size_t>("num_hidden_layers"),
                text_config->get<size_t>("max_position_embeddings"),
                text_config->get_dtype(),
                *static_kv_cache_config,
                rank_info_);
            break;
        }
        case backends::AttentionBackend::PAGED_ATTN: {
            auto paged_kv_cache_config = dynamic_cast<const cache::PagedKVCacheConfig *>(cache_config_.get());
            if (nullptr == paged_kv_cache_config) {
                throw std::runtime_error("paged_kv_cache_config is not initialized");
            }
            kv_cache_ = std::make_shared<cache::PagedKVCache>(
                text_config->get_head_dim(),
                text_config->get_head_dim(),
                text_config->get<size_t>("num_key_value_heads"),
                text_config->get<size_t>("num_key_value_heads"),
                text_config->get<size_t>("num_hidden_layers"),
                text_config->get_dtype(),
                *paged_kv_cache_config,
                rank_info_);
            break;
        }
        case backends::AttentionBackend::FLASH_ATTN: {
            auto flash_kv_cache_config = dynamic_cast<const cache::FlashKVCacheConfig *>(cache_config_.get());
            if (nullptr == flash_kv_cache_config) {
                throw std::runtime_error("flash_kv_cache_config is not initialized");
            }
            kv_cache_ = std::make_shared<cache::PagedKVCache>(
                text_config->get_head_dim(),
                text_config->get_head_dim(),
                text_config->get<size_t>("num_key_value_heads"),
                text_config->get<size_t>("num_key_value_heads"),
                text_config->get<size_t>("num_hidden_layers"),
                text_config->get_dtype(),
                *flash_kv_cache_config,
                rank_info_);
            break;
        }
        default:
            throw std::runtime_error("Unsupported attention backend: " + std::to_string(static_cast<int>(attention_backend)));
            break;
        };
    }

public:
    virtual ~InfinilmModel() = default;
    virtual Output forward(const Input &input) const = 0;
    virtual void reset_cache(const cache::CacheConfig *cache_config) {
        initialize_kv_cache(cache_config, model_config_);
    }

    const cache::CacheConfig *get_cache_config() const {
        return cache_config_.get();
    }
};
} // namespace infinilm
