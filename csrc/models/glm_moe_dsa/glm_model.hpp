#pragma once
#include "../../layers/linear/linear.hpp"
#include "../../layers/mlp/mlp.hpp"
#include "../deepseek_v2/deepseek_v2_mla_attention.hpp"
#include "../infinilm_model.hpp"
#include "glm_moe.hpp"
#include "glm_vocab_parallel.hpp"
#include "infinicore/nn/embedding.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include <infiniccl.h>
#include <vector>
namespace infinilm::models::glm_moe_dsa {
class GlmDecoder final : public infinicore::nn::Module {
public:
    GlmDecoder(std::shared_ptr<infinilm::config::ModelConfig>, size_t, const infinicore::Device &);
    void forward(const infinicore::Tensor &, infinicore::Tensor &, infinicore::Tensor &) const;

private:
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, input_layernorm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, post_attention_layernorm);
    INFINICORE_NN_MODULE(infinilm::models::deepseek_v2::DeepseekV2MLAAttention, self_attn);
    INFINICORE_NN_MODULE(GlmDenseMLP, dense_mlp);
    INFINICORE_NN_MODULE(GlmMoE, moe_mlp);
    bool moe_{false};
    size_t layer_idx_{0};
};
class GlmModel final : public infinicore::nn::Module {
public:
    GlmModel(std::shared_ptr<infinilm::config::ModelConfig>, const infinicore::Device &);
    infinicore::Tensor forward(const infinilm::InfinilmModel::Input &) const;
    void begin_pipeline_batch() const;

    size_t layer_start() const {
        return layer_start_;
    }
    size_t layer_end() const {
        return layer_end_;
    }

private:
    void transfer_pipeline_state_(size_t source_stage,
                                  infinicore::Tensor &hidden_states,
                                  infinicore::Tensor &residual,
                                  bool transfer_indexer_state) const;

    INFINICORE_NN_MODULE(GlmVocabEmbedding, embed_tokens);
    INFINICORE_NN_MODULE_VEC(GlmDecoder, layers);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm);
    size_t index_topk_{0};
    size_t hidden_size_{0};
    size_t layer_start_{0};
    size_t layer_end_{0};
    int pp_size_{1};
    int pp_rank_{0};
    infinicclComm_t pp_comm_{nullptr};
    std::vector<bool> stage_boundary_needs_topk_;
    mutable std::vector<infinicore::Tensor> pipeline_send_lifetimes_;
    infinicore::DataType dtype_;
};
class GlmForCausalLM final : public infinilm::InfinilmModel {
public:
    GlmForCausalLM(std::shared_ptr<infinilm::config::ModelConfig>, const infinicore::Device &);
    Output forward(const Input &) const override;
    void reset_cache(const cache::CacheConfig *) override;
    size_t max_decode_graph_batch_size() const override {
        return 16;
    }
    size_t decode_graph_batch_size(size_t batch_size) const override {
        for (const size_t bucket : {1UL, 2UL, 4UL, 8UL, 16UL}) {
            if (batch_size <= bucket) {
                return bucket;
            }
        }
        return batch_size;
    }

private:
    INFINICORE_NN_MODULE(GlmModel, model);
    INFINICORE_NN_MODULE(GlmVocabLMHead, lm_head);
    bool is_output_stage_{true};
    mutable std::vector<infinicore::Tensor> graph_microbatch_constants_;
};
std::shared_ptr<infinilm::config::ModelConfig> create_glm_config(std::shared_ptr<infinilm::config::ModelConfig>);
} // namespace infinilm::models::glm_moe_dsa
