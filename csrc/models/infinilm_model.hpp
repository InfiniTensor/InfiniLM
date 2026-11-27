#pragma once
#include <unordered_set>

#include "../backends/attention_backends.hpp"
#include "../cache/cache.hpp"
#include "../config/model_config.hpp"
#include "../layers/quantization/w4_model_plan.hpp"
#include "../weights/weights.hpp"
#include "runtime/tensor.hpp"

#include <optional>
#include <vector>

namespace infinilm {
namespace engine::distributed {
struct RankInfo;
}
namespace global_state {
struct ForwardContext;
}

class InfinilmModel : public infinilm::weights::ModelWeights {
public:
    struct Config {
        std::string model_type;
        virtual ~Config() = default;
    };

    struct Input {
        /// Token IDs: static `[batch, seq_len]`; paged packed `[1, token_count]`.
        std::optional<infinilm::runtime::Tensor> input_ids;
        /// Position IDs tensor of shape `[batch, seq_len]` or `[seq_len]`.
        std::optional<infinilm::runtime::Tensor> position_ids;
        /// Past Lengths of cached sequence for each request, of shape `[num_requests]`.
        std::optional<infinilm::runtime::Tensor> past_sequence_lengths;
        /// ToTal Lengths for each request sequence, of shape `[num_requests]`.
        std::optional<infinilm::runtime::Tensor> total_sequence_lengths;
        /// Offsets of each request in a continous-batched sequence, of shape `[num_requests + 1]`.
        std::optional<infinilm::runtime::Tensor> input_offsets;
        /// Cumulative total sequence lengths for each request, of shape `[num_requests + 1]`.
        std::optional<infinilm::runtime::Tensor> cu_seqlens;
        /// Block ids for each request `[batch, max_block_table_length]`. Used for paged cache.
        std::optional<infinilm::runtime::Tensor> block_tables;
        /// Slot ids for each token `[seq]`. Used for paged cache.
        std::optional<infinilm::runtime::Tensor> slot_mapping;
        /// Mamba state cache indices read at the start of each request forward, of shape `[num_requests]`.
        std::optional<infinilm::runtime::Tensor> mamba_init_state_indices;
        /// Mamba state cache indices written with the final state of each request forward, of shape `[num_requests]`.
        std::optional<infinilm::runtime::Tensor> mamba_final_state_indices;
        /// Image pixel values for multi-modal models.
        /// Vector of tensors. Shape is model-specific (e.g. LLaVA: [batch, 3, H, W], MiniCPM-V: [n_patch, 3, filter_H, H * W / filter_H]).
        std::optional<std::vector<infinilm::runtime::Tensor>> pixel_values;
        /// Image placeholder bounds for MiniCPM-V style replacement.
        /// Vector of tensors shape: [n_patch, 2].
        std::optional<std::vector<infinilm::runtime::Tensor>> image_bound;
        /// Target patch sizes for each image (MiniCPM-V).
        /// Vector of tensors shape: [n_path, 2] if pre-flattened.
        std::optional<std::vector<infinilm::runtime::Tensor>> tgt_sizes;
        /// Qwen-style image grids. Vector of tensors shape: [3] with temporal, height, width.
        std::optional<std::vector<infinilm::runtime::Tensor>> image_grid_thw;
        /// req_id for each pixel_values among a batch.
        std::optional<std::vector<size_t>> image_req_ids;
        /// Flattened [start, end) visual token ranges in the packed language sequence.
        std::optional<std::vector<size_t>> visual_token_ranges;
        /// Target model hidden states consumed by draft/MTP models.
        std::optional<infinilm::runtime::Tensor> target_hidden_states;
        /// Preserve logits for every packed position for speculative/MTP callers.
        bool sample_all_positions{false};
        size_t logical_kv_upper_bound{0};
        bool diffusion_preview{false};
        bool diffusion_expert_budget{false};
    };

    struct Output {
        /// Logits.
        infinilm::runtime::Tensor logits;
        /// Optional final hidden states, used by MTP/Eagle draft models.
        infinilm::runtime::Tensor hidden_states;
    };

    virtual ~InfinilmModel() = default;
    virtual Output forward(const Input &input,
                           global_state::ForwardContext &forward_context) const
        = 0;
    virtual void reset_cache(const cache::CacheConfig *cache_config,
                             global_state::ForwardContext &forward_context,
                             backends::AttentionBackend attention_backend,
                             const engine::distributed::RankInfo &rank);
    virtual const cache::CacheConfig *get_cache_config() const {
        return cache_config_.get();
    }
    const std::shared_ptr<infinilm::config::ModelConfig> &get_model_config() const {
        return model_config_;
    }

    // Opt in only when all request state touched by prepare lives in the
    // ordinary ForwardContext KV caches. Recurrent/vision/MLA models must
    // provide their own write-set contract before graph preparation is enabled.
    virtual bool supports_kv_only_prepare() const { return false; }

    void process_weights_after_loading(infinilm::runtime::Runtime &runtime, bool pre_transpose) override;
    // Validation consumes worker-owned values; no implicit rank/config snapshot.
    std::unique_ptr<quantization::W4ModelPlan> validate_w4_weights(
        infinilm::runtime::Runtime &runtime, bool pre_transpose,
        const engine::distributed::RankInfo &rank,
        const std::string &ep_backend, size_t ep_size);
    // Narrow owner revocation hook; never release backing before worker drain.
    virtual void prepare_paged_owner(global_state::ForwardContext &) const {}
    virtual void finish_paged_owner(bool) const {}
    virtual void invalidate_paged_owner() const noexcept {}

protected:
    void reset_default_cache(const cache::CacheConfig *cache_config,
                             global_state::ForwardContext &forward_context,
                             backends::AttentionBackend attention_backend,
                             const engine::distributed::RankInfo &rank);
    std::vector<infinilm::runtime::Tensor> default_allocate_kv_cache_tensors(
        infinilm::runtime::Runtime &runtime,
        const engine::distributed::RankInfo &rank,
        const cache::CacheConfig *cache_config,
        const std::shared_ptr<infinilm::config::ModelConfig> &text_config,
        const backends::AttentionBackend &attention_backend);

    std::unique_ptr<cache::CacheConfig> cache_config_;
    std::shared_ptr<infinilm::config::ModelConfig> model_config_;

private:
    static void process_weights_recursive_(infinilm::weights::ModelWeights *module, infinilm::runtime::Runtime &runtime, bool pre_transpose, std::unordered_set<const weights::ModelWeights *> &visited);
};
} // namespace infinilm
