#pragma once

#include "../../config/model_config.hpp"
#include "../../models/infinilm_model.hpp"
#include "kimi_k3_decoder_layer.hpp"

#include <infinicore/nn/embedding.hpp>
#include <infinicore/nn/module.hpp>
#include <infinicore/nn/rmsnorm.hpp>
#include <infinicore/tensor.hpp>

#include <memory>

namespace infinilm::models::kimi_k3 {

class KimiK3TextModel : public infinicore::nn::Module {
public:
    KimiK3TextModel(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                    const infinicore::Device &device);

    infinicore::Tensor forward(const infinilm::InfinilmModel::Input &input) const;
    infinicore::Tensor forward_embeds(const infinicore::Tensor &inputs_embeds) const;
    infinicore::Tensor embed_tokens(const infinicore::Tensor &input_ids) const;
    bool is_first_pp_stage() const { return pp_stage_ == 0; }
    bool is_last_pp_stage() const { return pp_stage_ + 1 == pp_size_; }

private:
    struct PipelineState {
        infinicore::Tensor hidden_states;
        infinicore::Tensor block_residual;
    };

    PipelineState initial_state(const infinicore::Tensor &first_stage_hidden) const;
    void send_pipeline_state(const PipelineState &state) const;
    infinicore::Tensor recv_sharded_last_dim(const infinicore::Shape &shape) const;
    void send_sharded_last_dim(const infinicore::Tensor &tensor) const;
    infinicore::Tensor apply_output_attn_res(const infinicore::Tensor &hidden_states,
                                             const infinicore::Tensor &block_residual) const;

    INFINICORE_NN_MODULE(infinicore::nn::Embedding, embed_tokens);
    INFINICORE_NN_MODULE_VEC(KimiK3DecoderLayer, layers);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, output_attn_res_norm);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, output_attn_res_proj);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm);

    infinicore::DataType dtype_{infinicore::DataType::F32};
    size_t hidden_size_{0};
    size_t attn_res_block_size_{12};
    size_t pp_size_{1};
    size_t pp_stage_{0};
    size_t tp_size_{1};
    size_t tp_rank_{0};
    size_t local_layer_begin_{0};
    size_t local_layer_end_{0};
    infinicore::Device device_;
};

} // namespace infinilm::models::kimi_k3
