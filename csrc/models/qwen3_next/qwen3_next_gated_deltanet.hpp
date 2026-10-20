#pragma once
#include "../../layers/normalization/rmsnorm.hpp"
#include "../../weights/weights.hpp"

#include "../../layers/common_modules.hpp"

namespace infinilm::global_state {
struct ForwardContext;
}

namespace infinilm::models::qwen3_next {
class Qwen3NextCausalConv1D : public infinilm::weights::ModelWeights {
protected:
    infinilm::runtime::Device device_;

public:
    Qwen3NextCausalConv1D(infinilm::runtime::Runtime &runtime,
                          std::shared_ptr<infinilm::config::ModelConfig> model_config,
                          size_t layer_idx,
                          const engine::distributed::RankInfo &rank_info,
                          backends::AttentionBackend attention_backend,
                          const layers::moe::EPConfig &ep_config);

    infinilm::runtime::Tensor forward(const infinilm::runtime::Tensor &qkv,
                                      const infinilm::global_state::ForwardContext &forward_context) const;
    void process_weights_after_loading(infinilm::runtime::Runtime &runtime, bool pre_transpose) override;

private:
    infinilm::weights::Weight weight_;
    infinilm::runtime::DataType weight_dtype_;
    bool postload_failed_ = false;

    size_t layer_idx_;
    size_t local_conv_dim_;
    size_t full_qk_dim_;
    size_t full_v_dim_;
    size_t local_qk_dim_;
    size_t local_v_dim_;
    size_t conv_kernel_dim_;
    size_t tp_rank_;
    size_t tp_size_;
};

class Qwen3NextGatedDeltaNet : public infinilm::weights::ModelWeights {
protected:
    infinilm::runtime::Device device_;

public:
    std::vector<std::shared_ptr<infinilm::weights::ModelWeights>> loading_children() const override {
        auto result = ModelWeights::loading_children();
        if (in_proj_qkv_) {
            result.push_back(in_proj_qkv_);
        }
        return result;
    }

    Qwen3NextGatedDeltaNet(infinilm::runtime::Runtime &runtime,
                           std::shared_ptr<infinilm::config::ModelConfig> model_config,
                           size_t layer_idx,
                           const engine::distributed::RankInfo &rank_info,
                           backends::AttentionBackend attention_backend,
                           const layers::moe::EPConfig &ep_config);

    infinilm::runtime::Tensor forward(const infinilm::runtime::Tensor &hidden_states,
                                      const infinilm::global_state::ForwardContext &forward_context) const;

private:
    std::shared_ptr<layers::linear::QKVParallelLinear> in_proj_qkv_;
    std::shared_ptr<layers::linear::ColumnParallelLinear> in_proj_z_;
    std::shared_ptr<layers::linear::ColumnParallelLinear> in_proj_a_;
    std::shared_ptr<layers::linear::ColumnParallelLinear> in_proj_b_;
    std::shared_ptr<Qwen3NextCausalConv1D> conv1d_;

    infinilm::weights::Weight dt_bias_;
    infinilm::weights::Weight A_log_;
    std::shared_ptr<infinilm::layers::RMSNorm> norm_;
    std::shared_ptr<layers::linear::RowParallelLinear> out_proj_;

    size_t layer_idx_;
    size_t local_num_value_heads_;
    size_t local_num_key_heads_;
    size_t key_head_dim_;
    size_t value_head_dim_;
    size_t local_key_dim_;
    size_t local_value_dim_;
};

} // namespace infinilm::models::qwen3_next
