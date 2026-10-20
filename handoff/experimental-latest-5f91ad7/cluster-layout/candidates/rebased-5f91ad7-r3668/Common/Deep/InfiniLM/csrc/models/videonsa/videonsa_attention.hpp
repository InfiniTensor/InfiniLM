#pragma once
#include "nsa_serial_lane.hpp"
#include <cstdint>

#include "../../layers/attention/attention.hpp"
#include <array>

#include <optional>

namespace infinilm::models::videonsa {

class VideoNSAAttention : public infinilm::layers::attention::Attention {
public:
    VideoNSAAttention(infinilm::runtime::Runtime &runtime,
                      std::shared_ptr<infinilm::config::ModelConfig> model_config,
                      size_t layer_idx, const engine::distributed::RankInfo &rank,
                      backends::AttentionBackend attention_backend,
                      const layers::moe::EPConfig &ep);

    infinilm::runtime::Tensor forward(const infinilm::runtime::Tensor &positions,
                                      const infinilm::runtime::Tensor &hidden_states,
                                      global_state::ForwardContext &forward_context) const;

    void prepare_paged_owner(global_state::ForwardContext &) const;
    void finish_paged_owner(bool) const;
    void invalidate_paged_owner() const noexcept;

protected:
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> g_proj_1_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> g_proj_2_;

    mutable std::optional<infinilm::runtime::Tensor> nsa_k_cmp_cache_;
    mutable std::optional<infinilm::runtime::Tensor> nsa_v_cmp_cache_;
    mutable nsa_serial::Lane nsa_lane_;
    mutable std::optional<infinilm::runtime::Tensor> nsa_raw_lease_;
    mutable std::uint64_t nsa_backing_epoch_{0}, nsa_compressed_generation_{0}, nsa_prepared_invocation_{0};
    mutable bool nsa_prepared_tail_{false}, nsa_update_last_only_{false};
    std::shared_ptr<infinilm::runtime::Runtime> mrope_cache_runtime_owner_;
    mutable std::optional<infinilm::runtime::Tensor> mrope_sin_cache_;
    mutable std::optional<infinilm::runtime::Tensor> mrope_cos_cache_;
    std::array<int, 3> mrope_section_ = {0, 0, 0};
    size_t mrope_rotary_dim_ = 0;
    bool mrope_interleaved_ = false;
    size_t tp_rank_ = 0;
    size_t total_num_attention_heads_ = 0;
    size_t max_position_embeddings_ = 0;
};

} // namespace infinilm::models::videonsa
