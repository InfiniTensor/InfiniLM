#pragma once

#include "../../config/model_config.hpp"
#include "../../models/infinilm_model.hpp"
#include "kimi_k3_decoder_layer.hpp"

#include "../../layers/embedding/embedding.hpp"
#include "../../layers/normalization/rmsnorm.hpp"
#include "../../weights/weights.hpp"
#include <infiniccl/infiniccl.h>
#include <runtime/tensor.hpp>

#include <memory>

namespace infinilm::global_state {
struct ForwardContext;
}

namespace infinilm::models::kimi_k3 {

class KimiK3TextModel : public infinilm::weights::ModelWeights {
public:
    KimiK3TextModel(
        infinilm::runtime::Runtime &runtime,
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
        const engine::distributed::RankInfo &rank_info,
        backends::AttentionBackend attention_backend,
        const layers::moe::EPConfig &ep_config);

    infinilm::runtime::Tensor forward(const infinilm::InfinilmModel::Input &input,
                                      global_state::ForwardContext &forward_context) const;
    infinilm::runtime::Tensor forward_embeds(const infinilm::runtime::Tensor &inputs_embeds,
                                             global_state::ForwardContext &forward_context) const;
    infinilm::runtime::Tensor embed_tokens(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &input_ids) const;
    bool is_first_pp_stage() const { return pp_stage_ == 0; }
    bool is_last_pp_stage() const { return pp_stage_ + 1 == pp_size_; }

private:
    struct PipelineState {
        infinilm::runtime::Tensor hidden_states;
        infinilm::runtime::Tensor block_residual_storage;
        size_t block_residual_count;
    };

    PipelineState initial_state(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &first_stage_hidden) const;
    void send_pipeline_state(infinilm::runtime::Runtime &runtime, const PipelineState &state) const;
    infinilm::runtime::Tensor recv_sharded_last_dim(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Shape &shape) const;
    infinilm::runtime::Tensor apply_output_attn_res(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &hidden_states,
                                                    const infinilm::runtime::Tensor &block_residual_storage,
                                                    size_t block_residual_count) const;

    std::shared_ptr<infinilm::layers::Embedding> embed_tokens_;
    std::vector<std::shared_ptr<KimiK3DecoderLayer>> layers_;
    std::shared_ptr<infinilm::layers::RMSNorm> output_attn_res_norm_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> output_attn_res_proj_;
    std::shared_ptr<infinilm::layers::RMSNorm> norm_;

    infinilm::runtime::DataType dtype_{infinilm::runtime::DataType::kFloat32};
    size_t hidden_size_{0};
    size_t attn_res_block_size_{12};
    size_t pp_size_{1};
    size_t pp_stage_{0};
    size_t tp_size_{1};
    size_t tp_rank_{0};
    // Borrowed from CommunicationGroup; the model/worker are destroyed before the group.
    infinicclComm_t tp_comm_{nullptr};
    infinicclComm_t world_comm_{nullptr};
    size_t local_layer_begin_{0};
    size_t local_layer_end_{0};
    size_t block_residual_capacity_{0};
    infinilm::runtime::Device device_;
};

} // namespace infinilm::models::kimi_k3
