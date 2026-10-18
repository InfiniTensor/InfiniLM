#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/common_modules.hpp"
#include "../../layers/linear/linear.hpp"
#include "../../layers/mlp/mlp.hpp"
#include "../../layers/moe/legacy/moe_mlp.hpp"
#include "../../weights/weights.hpp"
#include "runtime/device.hpp"
#include "runtime/tensor.hpp"
#include <infiniccl/infiniccl.h>

#include <memory>
#include <tuple>
#include <vector>

namespace infinilm::models::deepseek_v2 {

using DeepseekV2MLP = infinilm::layers::mlp::MLP;
using DeepseekV2ExpertMLP = infinilm::layers::moe::legacy::MoeMLP;

class DeepseekV2TopKRouter : public infinilm::weights::ModelWeights {
public:
    DeepseekV2TopKRouter(infinilm::runtime::Runtime &runtime,
                         std::shared_ptr<infinilm::config::ModelConfig> model_config);

    std::tuple<infinilm::runtime::Tensor, infinilm::runtime::Tensor> forward(
        infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &hidden_states) const;

protected:
    infinilm::weights::Weight weight_;
    size_t num_experts_per_tok_{0};
    size_t num_experts_{0};
    bool norm_topk_prob_{false};
};

class DeepseekV2Experts : public infinilm::weights::ModelWeights {
public:
    DeepseekV2Experts(infinilm::runtime::Runtime &runtime,
                      std::shared_ptr<infinilm::config::ModelConfig> model_config,
                      const engine::distributed::RankInfo &rank_info);

    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime,
                                      const infinilm::runtime::Tensor &hidden_states,
                                      const infinilm::runtime::Tensor &top_k_index,
                                      const infinilm::runtime::Tensor &top_k_weights) const;

protected:
    infinilm::runtime::Tensor forward_cpu_routed_(infinilm::runtime::Runtime &runtime,
                                                  const infinilm::runtime::Tensor &hidden_states,
                                                  const infinilm::runtime::Tensor &top_k_index,
                                                  const infinilm::runtime::Tensor &top_k_weights) const;

    std::vector<std::shared_ptr<DeepseekV2ExpertMLP>> experts_;
    std::vector<infinilm::runtime::Tensor> gate_weights_;
    std::vector<infinilm::runtime::Tensor> up_weights_;
    std::vector<infinilm::runtime::Tensor> down_weights_;
    size_t hidden_size_{0};
    size_t moe_intermediate_size_{0};
    size_t local_moe_intermediate_size_{0};
    size_t num_experts_per_tok_{0};
    size_t num_experts_{0};
    size_t tp_size_{1};
    infinicclComm_t communicator_{nullptr};
};

class DeepseekV2MoE : public infinilm::weights::ModelWeights {
public:
    DeepseekV2MoE(infinilm::runtime::Runtime &runtime,
                  std::shared_ptr<infinilm::config::ModelConfig> model_config,
                  size_t layer_idx,
                  const engine::distributed::RankInfo &rank_info,
                  backends::AttentionBackend attention_backend,
                  const layers::moe::EPConfig &ep_config);

    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &hidden_states) const;

protected:
    std::shared_ptr<DeepseekV2TopKRouter> gate_;
    std::shared_ptr<DeepseekV2Experts> experts_;
    std::shared_ptr<DeepseekV2MLP> shared_experts_;
    bool has_shared_experts_{false};
};

} // namespace infinilm::models::deepseek_v2
