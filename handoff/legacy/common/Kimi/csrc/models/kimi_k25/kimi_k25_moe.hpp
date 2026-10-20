#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/mlp/mlp.hpp"
#include "../../layers/moe/common/moe_types.hpp"
#include "../../layers/moe/experts/fused_moe_experts.hpp"
#include "../../layers/moe/fused_moe.hpp"
#include "../../layers/moe/router/topk_router.hpp"

#include "../../weights/weights.hpp"
#include <runtime/tensor.hpp>

#include <cstddef>
#include <memory>
#include <vector>

namespace infinilm::models::kimi_k25 {

class KimiK25MXFP4Experts : public infinilm::weights::ModelWeights {
public:
    KimiK25MXFP4Experts(
    infinilm::runtime::Runtime &runtime,
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const engine::distributed::RankInfo &rank_info);

    const infinilm::layers::moe::MoeWeights &moe_weights() const;
    void process_weights_after_loading(infinilm::runtime::Runtime &runtime, bool pre_transpose) override;

protected:
    std::vector<infinilm::weights::Weight> packed_parameters_;
    infinilm::layers::moe::MoeWeights moe_weights_;
    size_t num_experts_{0};
    size_t hidden_size_{0};
    size_t local_intermediate_size_{0};
    infinilm::runtime::DataType dtype_;
    infinilm::runtime::Device device_;
};

class KimiK25MoE : public infinilm::weights::ModelWeights {
public:
    KimiK25MoE(
    infinilm::runtime::Runtime &runtime,
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
               size_t layer_idx,
    const engine::distributed::RankInfo &rank_info,
    backends::AttentionBackend attention_backend,
    const layers::moe::EPConfig &ep_config);

    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &hidden_states) const;

protected:
    std::shared_ptr<infinilm::layers::moe::TopKRouter> gate_;
    std::shared_ptr<infinilm::layers::moe::FusedMoeExperts> experts_;
    std::shared_ptr<KimiK25MXFP4Experts> mxfp4_experts_;
    std::shared_ptr<infinilm::layers::moe::FusedMoE> fused_moe_;
    std::shared_ptr<infinilm::layers::mlp::MLP> shared_experts_;
    bool uses_mxfp4_experts_{false};
    float routed_scaling_factor_{1.0f};
};

std::shared_ptr<infinilm::config::ModelConfig>
make_kimi_mlp_config(const std::shared_ptr<infinilm::config::ModelConfig> &model_config,
                     size_t intermediate_size);

} // namespace infinilm::models::kimi_k25
