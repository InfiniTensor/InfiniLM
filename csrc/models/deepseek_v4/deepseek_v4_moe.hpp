#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/quantization/base_quantization.hpp"
#include "../../layers/linear/linear.hpp"
#include "../../weights/weights.hpp"
#include <infiniccl/infiniccl.h>

namespace infinilm::global_state { struct ForwardContext; }
namespace infinilm::engine::distributed { struct RankInfo; }
namespace infinilm::backends { enum class AttentionBackend; }
namespace infinilm::layers::moe { struct EPConfig; }

namespace infinilm::models::deepseek_v4 {

class DeepseekV4ExpertBank final : public infinilm::weights::ModelWeights {
public:
    DeepseekV4ExpertBank(infinilm::runtime::Runtime &runtime,
                         const std::shared_ptr<infinilm::config::ModelConfig> &model_config,
                         const engine::distributed::RankInfo &rank_info,
                         size_t expert_count,
                         size_t intermediate_size,
                         bool grouped_layout);

    infinilm::runtime::Tensor slice(const infinilm::weights::Weight &weight, size_t expert_index) const;
    size_t expert_count() const { return expert_count_; }
    void process_weights_after_loading(
        infinilm::runtime::Runtime &runtime, bool pre_transpose) override;
private:
    size_t expert_count_{0};
    size_t local_intermediate_size_{0};
    bool grouped_layout_{false};
    bool fused_w13_{true};

    // Routed experts are loaded directly into the ACLNN grouped-matmul layout.
    infinilm::weights::Weight w13_bank_;
    infinilm::weights::Weight w13_scale_bank_;
    infinilm::weights::Weight grouped_w2_bank_;
    infinilm::weights::Weight grouped_w2_scale_bank_;

    // The single shared expert keeps the regular linear layout.
    infinilm::weights::Weight w1_bank_;
    infinilm::weights::Weight w1_scale_bank_;
    infinilm::weights::Weight w2_bank_;
    infinilm::weights::Weight w2_scale_bank_;
    infinilm::weights::Weight w3_bank_;
    infinilm::weights::Weight w3_scale_bank_;
    friend class DeepseekV4Expert;
    friend class DeepseekV4MoE;
};

class DeepseekV4Expert final : public infinilm::weights::ModelWeights {
public:
    DeepseekV4Expert(const std::shared_ptr<DeepseekV4ExpertBank> &bank,
                     size_t expert_index,
                     const std::shared_ptr<infinilm::config::ModelConfig> &model_config,
                     const engine::distributed::RankInfo &rank_info);

    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime,
                                      const infinilm::runtime::Tensor &hidden_states,
                                      bool allreduce = true) const;

private:
    std::shared_ptr<infinilm::quantization::BaseQuantization> quantization_;
    infinicclComm_t communicator_{nullptr};
    infinilm::weights::Weight w1_;
    infinilm::weights::Weight w1_scale_;
    infinilm::weights::Weight w2_;
    infinilm::weights::Weight w2_scale_;
    infinilm::weights::Weight w3_;
    infinilm::weights::Weight w3_scale_;
};

class DeepseekV4MoE final : public infinilm::weights::ModelWeights {
public:
    DeepseekV4MoE(infinilm::runtime::Runtime &runtime,
                  std::shared_ptr<infinilm::config::ModelConfig> model_config,
                  size_t layer_idx,
                  const engine::distributed::RankInfo &rank_info,
                  backends::AttentionBackend attention_backend,
                  const layers::moe::EPConfig &ep_config);

    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime,
                                      const infinilm::runtime::Tensor &hidden_states,
                                      const infinilm::runtime::Tensor &input_ids,
                                      infinilm::global_state::ForwardContext &forward_context) const;

private:
    size_t layer_idx_{0};
    size_t hidden_size_{0};
    size_t local_intermediate_size_{0};
    size_t expert_count_{0};
    size_t topk_{0};
    double routed_scaling_factor_{1.0};
    double swiglu_limit_{10.0};
    bool hash_router_{false};
    bool fused_w13_{true};
    infinicclComm_t communicator_{nullptr};
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> gate_;
    infinilm::weights::Weight gate_bias_;
    infinilm::weights::Weight tid2eid_;
    std::shared_ptr<DeepseekV4ExpertBank> expert_bank_;
    std::vector<std::shared_ptr<DeepseekV4Expert>> experts_;
    std::shared_ptr<DeepseekV4ExpertBank> shared_expert_bank_;
    std::shared_ptr<DeepseekV4Expert> shared_experts_;
};

} // namespace infinilm::models::deepseek_v4
