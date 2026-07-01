#pragma once

#include "../backends/attention_backends.hpp"
#include "../cache/cache.hpp"
#include "../config/model_config.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"

#include <optional>
#include <vector>

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
        /// Image pixel values for multi-modal models.
        /// Vector of tensors. Shape is model-specific (e.g. LLaVA: [batch, 3, H, W], MiniCPM-V: [n_patch, 3, filter_H, H * W / filter_H]).
        std::optional<std::vector<infinicore::Tensor>> pixel_values;
        /// Image placeholder bounds for MiniCPM-V style replacement.
        /// Vector of tensors shape: [n_patch, 2].
        std::optional<std::vector<infinicore::Tensor>> image_bound;
        /// Target patch sizes for each image (MiniCPM-V).
        /// Vector of tensors shape: [n_path, 2] if pre-flattened.
        std::optional<std::vector<infinicore::Tensor>> tgt_sizes;
        /// Flattened [start, end) visual token ranges in the packed language sequence.
        std::optional<std::vector<size_t>> visual_token_ranges;
    };

    struct Output {
        /// Logits.
        infinicore::Tensor logits;
    };

    virtual ~InfinilmModel() = default;
    virtual Output forward(const Input &input) const = 0;
    virtual void reset_cache(const cache::CacheConfig *cache_config);
    virtual const cache::CacheConfig *get_cache_config() const {
        return cache_config_.get();
    }

    void process_weights_after_loading();
    void reset_runtime_state() const;

protected:
    std::vector<infinicore::Tensor> default_allocate_kv_cache_tensors(
        const cache::CacheConfig *cache_config,
        const std::shared_ptr<infinilm::config::ModelConfig> &text_config,
        const backends::AttentionBackend &attention_backend);

    std::unique_ptr<cache::CacheConfig> cache_config_;
    std::shared_ptr<infinilm::config::ModelConfig> model_config_;

private:
    static void process_weights_recursive_(infinicore::nn::Module *module);
    static void reset_runtime_state_recursive_(const infinicore::nn::Module *module);
};
} // namespace infinilm
