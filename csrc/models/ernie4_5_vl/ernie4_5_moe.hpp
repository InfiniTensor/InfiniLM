#pragma once

#include "../../layers/common_modules.hpp"
#include "../../layers/linear/linear.hpp"
#include "../../layers/moe/runner/cuda_fused_moe_runner.hpp"
#include "../../weights/weights.hpp"
#include "runtime/tensor.hpp"

#include <array>
#include <memory>
#include <tuple>
#include <vector>

namespace infinilm::models::ernie4_5_vl {

class Ernie45TopKRouter : public infinilm::weights::ModelWeights {
public:
    Ernie45TopKRouter(infinilm::runtime::Runtime &runtime,
        std::shared_ptr<infinilm::config::ModelConfig> model_config);

    std::tuple<infinilm::runtime::Tensor, infinilm::runtime::Tensor> forward(infinilm::runtime::Runtime &runtime,
                                                               const infinilm::runtime::Tensor &hidden_states,
                                                               const infinilm::runtime::Tensor &correction_bias = infinilm::runtime::Tensor(),
                                                               size_t group_idx = 0) const;

protected:
    infinilm::weights::Weight weight_;
    infinilm::weights::Weight weight_1_;
    size_t num_experts_per_tok_{0};
    bool norm_topk_prob_{false};
};

class Ernie45ExpertMLP : public infinilm::weights::ModelWeights {
public:
    Ernie45ExpertMLP(infinilm::runtime::Runtime &runtime,
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
                     size_t intermediate_size,
        const engine::distributed::RankInfo &rank_info);

    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime,
                               const infinilm::runtime::Tensor &hidden_states) const;
    void set_alpha(float alpha) { down_proj_->set_alpha(alpha); }

protected:
    std::shared_ptr<infinilm::layers::linear::ColumnParallelLinear> gate_proj_;
    std::shared_ptr<infinilm::layers::linear::ColumnParallelLinear> up_proj_;
    std::shared_ptr<infinilm::layers::linear::RowParallelLinear> down_proj_;
};

class Ernie45Experts : public infinilm::weights::ModelWeights {
public:
    Ernie45Experts(infinilm::runtime::Runtime &runtime,
        std::shared_ptr<infinilm::config::ModelConfig> model_config);

    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime,
                               const infinilm::runtime::Tensor &hidden_states,
                               const infinilm::runtime::Tensor &top_k_index,
                               const infinilm::runtime::Tensor &top_k_weights,
                               size_t group_idx = 0) const;

protected:
    infinilm::weights::Weight w1_;
    infinilm::weights::Weight w2_;
    infinilm::weights::Weight b1_;
    infinilm::weights::Weight b2_;
    infinilm::weights::Weight w1_1_;
    infinilm::weights::Weight w2_1_;
    infinilm::weights::Weight b1_1_;
    infinilm::weights::Weight b2_1_;
    size_t num_experts_per_tok_{0};
    size_t num_text_experts_{0};
    size_t num_vision_experts_{0};
    bool use_bias_{false};
    // The workspace is per group and may only be used by this Runtime owner.
    std::shared_ptr<infinilm::runtime::Runtime> runtime_owner_;
    std::array<size_t, 2> intermediate_sizes_{};
    std::array<std::unique_ptr<layers::moe::CudaFusedMoeRunner>, 2> runners_;
    mutable std::array<layers::moe::MoeWorkspace, 2> workspaces_;
};

class Ernie45MoE : public infinilm::weights::ModelWeights {
public:
    Ernie45MoE(infinilm::runtime::Runtime &runtime,
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
        size_t layer_idx,
        const engine::distributed::RankInfo &rank_info,
        backends::AttentionBackend attention_backend,
        const layers::moe::EPConfig &ep_config);

    infinilm::runtime::Tensor forward(const infinilm::runtime::Tensor &hidden_states,
                               global_state::ForwardContext &forward_context) const;

protected:
    infinilm::runtime::Tensor forward_group(infinilm::runtime::Runtime &runtime,
                                     const infinilm::runtime::Tensor &hidden_states_2d, size_t group_idx) const;

    std::shared_ptr<Ernie45TopKRouter> gate_;
    std::shared_ptr<Ernie45Experts> experts_;
    std::shared_ptr<infinilm::layers::mlp::MLP> shared_experts_;
    infinilm::weights::Weight e_score_correction_bias_;
    size_t num_experts_{0};
    size_t num_vision_experts_{0};
    bool has_shared_experts_{false};
};

} // namespace infinilm::models::ernie4_5_vl
