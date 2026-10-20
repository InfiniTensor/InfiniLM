#pragma once

#include "../../backends/attention_backends.hpp"
#include "../../config/model_config.hpp"
#include "../../layers/linear/linear.hpp"
#include "../../layers/moe/common/moe_types.hpp"
#include "../../layers/moe/router/topk_router.hpp"

#include "../../layers/normalization/rmsnorm.hpp"
#include "../../weights/weights.hpp"
#include <infiniccl/infiniccl.h>
#include <runtime/tensor.hpp>

#include <memory>

namespace infinilm::models::kimi_k3 {

struct KimiK3Mxfp4MoeWeights {
    infinilm::runtime::Tensor packed_w13;
    infinilm::runtime::Tensor w13_scale;
    infinilm::runtime::Tensor packed_w2;
    infinilm::runtime::Tensor w2_scale;
};

class KimiK3MLP : public infinilm::weights::ModelWeights {
public:
    std::vector<std::shared_ptr<infinilm::weights::ModelWeights>> loading_children() const override {
        auto result = ModelWeights::loading_children();
        if (gate_up_proj_) {
            result.push_back(gate_up_proj_);
        }
        return result;
    }

    KimiK3MLP(
        infinilm::runtime::Runtime &runtime,
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
        size_t intermediate_size,
        const engine::distributed::RankInfo &rank_info);

    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &hidden_states) const;

private:
    std::shared_ptr<infinilm::layers::linear::GateUpParallelLinear> gate_up_proj_;
    std::shared_ptr<infinilm::layers::linear::RowParallelLinear> down_proj_;
    float situ_beta_{4.0f};
    float situ_linear_beta_{25.0f};
    infinilm::runtime::Device device_;
};

class KimiK3Experts : public infinilm::weights::ModelWeights {
public:
    KimiK3Experts(
        infinilm::runtime::Runtime &runtime,
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
        const engine::distributed::RankInfo &rank_info);

    const KimiK3Mxfp4MoeWeights &mxfp4_weights() const;

private:
    void register_mxfp4_experts(infinilm::runtime::Runtime &runtime);

    KimiK3Mxfp4MoeWeights mxfp4_weights_;
    size_t num_experts_{0};
    size_t hidden_size_{0};
    size_t local_intermediate_size_{0};
    size_t tp_rank_{0};
    size_t tp_size_{1};
    infinilm::runtime::Device device_;
};

class KimiK3MoE : public infinilm::weights::ModelWeights {
public:
    KimiK3MoE(
        infinilm::runtime::Runtime &runtime,
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
        size_t layer_idx,
        const engine::distributed::RankInfo &rank_info,
        backends::AttentionBackend attention_backend,
        const layers::moe::EPConfig &ep_config);

    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &hidden_states) const;

private:
    std::shared_ptr<infinilm::layers::moe::TopKRouter> gate_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> routed_expert_down_proj_;
    std::shared_ptr<KimiK3Experts> experts_;
    std::shared_ptr<infinilm::layers::RMSNorm> routed_expert_norm_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> routed_expert_up_proj_;
    std::shared_ptr<KimiK3MLP> shared_experts_;
    mutable infinilm::runtime::Tensor tiled_scratch_r1861_;
    int tp_size_{1};
    infinicclComm_t communicator_{nullptr};
};

std::shared_ptr<infinilm::config::ModelConfig>
make_kimi_k3_subconfig(const std::shared_ptr<infinilm::config::ModelConfig> &model_config,
                       size_t hidden_size,
                       size_t intermediate_size);

} // namespace infinilm::models::kimi_k3
