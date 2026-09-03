#pragma once

#include "../../models/infinilm_model.hpp"
#include "minimax_text_01_decoder_layer.hpp"

#include <infinicore/nn/embedding.hpp>
#include <infinicore/nn/module.hpp>
#include <infinicore/nn/rmsnorm.hpp>
#include <infinicore/tensor.hpp>

#include <cstddef>
#include <memory>
#include <vector>

namespace infinilm::models::minimax_text_01 {

/**
 * @brief The transformer body of MiniMax-Text-01.
 *
 * Embedding -> decoder layers -> final norm. A dedicated loop is used instead
 * of the `TextModel` template because MiniMax blocks apply their own post-norm
 * alpha/beta residual combination and therefore do not satisfy the fused
 * residual-stream contract expected by `TextModel`.
 */
class MiniMaxText01Model : public infinicore::nn::Module {
public:
    MiniMaxText01Model(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                       const infinicore::Device &device);

    infinicore::Tensor forward(const infinilm::InfinilmModel::Input &input) const;

private:
    INFINICORE_NN_MODULE(infinicore::nn::Embedding, embed_tokens);
    INFINICORE_NN_MODULE_VEC(MiniMaxText01DecoderLayer, layers);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm);

    bool is_first_pp_stage() const { return pp_stage_ == 0; }
    bool is_last_pp_stage() const { return pp_stage_ + 1 == pp_size_; }

    infinicore::Tensor initial_hidden_states(const infinilm::InfinilmModel::Input &input) const;
    infinicore::Tensor recv_pipeline_hidden(size_t batch_size,
                                            size_t seq_len,
                                            const infinicore::DataType &dtype_hint,
                                            const infinicore::Device &device) const;
    void send_pipeline_hidden(const infinicore::Tensor &hidden_states) const;

    infinicore::DataType dtype_{infinicore::DataType::F32};
    size_t hidden_size_{0};
    size_t pp_size_{1};
    size_t pp_stage_{0};
    size_t tp_size_{1};
    size_t tp_rank_{0};
    size_t local_layer_begin_{0};
    size_t local_layer_end_{0};
};

/**
 * @brief Top-level causal LM for MiniMax-Text-01.
 *
 * A dedicated class instead of the `TextCausalLM` template so that
 * `reset_cache` can allocate the hybrid KV / Lightning recurrent state caches.
 */
class MiniMaxText01ForCausalLM : public InfinilmModel {
public:
    MiniMaxText01ForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                             const infinicore::Device &device);

    Output forward(const Input &input) const override;

    void reset_cache(const cache::CacheConfig *cache_config) override;

protected:
    INFINICORE_NN_MODULE(MiniMaxText01Model, model);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, lm_head);

private:
    bool is_last_pp_stage() const { return pp_stage_ + 1 == pp_size_; }

    size_t pp_size_{1};
    size_t pp_stage_{0};
};

std::shared_ptr<infinilm::config::ModelConfig>
create_minimax_text_01_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::minimax_text_01
