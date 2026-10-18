#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/linear/fused_linear.hpp"
#include "../../layers/linear/linear.hpp"

#include "../../weights/weights.hpp"
#include "../../layers/normalization/rmsnorm.hpp"
#include <runtime/tensor.hpp>

#include <memory>

namespace infinilm::global_state { struct ForwardContext; }

namespace infinilm::models::kimi_k3 {

class KimiK3ShortConv : public infinilm::weights::ModelWeights {
protected:
    infinilm::runtime::Device device_;

public:
    KimiK3ShortConv(infinilm::runtime::Runtime &runtime,
                    size_t full_channels,
                    size_t kernel_size,
                    const infinilm::runtime::DataType &dtype,
                    const engine::distributed::RankInfo &rank_info);

    infinilm::runtime::Tensor weight() const { return weight_; }

private:
    infinilm::weights::Weight weight_;
};

class KimiK3DeltaAttention : public infinilm::weights::ModelWeights {
protected:
    infinilm::runtime::Device device_;

public:
    std::vector<std::shared_ptr<infinilm::weights::ModelWeights>> loading_children() const override {
        auto result = ModelWeights::loading_children();
        if (qkv_proj_) result.push_back(qkv_proj_);
        return result;
    }

    KimiK3DeltaAttention(infinilm::runtime::Runtime &runtime,
                         std::shared_ptr<infinilm::config::ModelConfig> model_config,
                         size_t layer_idx,
                         const engine::distributed::RankInfo &rank_info);

    infinilm::runtime::Tensor forward(const infinilm::runtime::Tensor &hidden_states,
                              global_state::ForwardContext &forward_context) const;
    void process_weights_after_loading(infinilm::runtime::Runtime &runtime, bool pre_transpose) override;

private:
    size_t layer_idx_{0};
    size_t local_num_heads_{0};
    size_t head_dim_{0};
    size_t local_projection_size_{0};
    size_t conv_kernel_size_{0};
    infinilm::runtime::DataType conv_dtype_;
    bool postload_attempted_{false};
    float gate_lower_bound_{-5.0f};

    std::shared_ptr<infinilm::layers::linear::QKVParallelLinear> qkv_proj_;
    std::shared_ptr<KimiK3ShortConv> q_conv1d_;
    std::shared_ptr<KimiK3ShortConv> k_conv1d_;
    std::shared_ptr<KimiK3ShortConv> v_conv1d_;
    infinilm::runtime::Tensor packed_conv_weight_;
    infinilm::weights::Weight A_log_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> f_a_proj_;
    std::shared_ptr<infinilm::layers::linear::ColumnParallelLinear> f_b_proj_;
    infinilm::weights::Weight dt_bias_;
    std::shared_ptr<infinilm::layers::linear::ColumnParallelLinear> b_proj_;
    std::shared_ptr<infinilm::layers::linear::ColumnParallelLinear> g_proj_;
    std::shared_ptr<infinilm::layers::RMSNorm> o_norm_;
    std::shared_ptr<infinilm::layers::linear::RowParallelLinear> o_proj_;
};

} // namespace infinilm::models::kimi_k3
