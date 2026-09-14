#pragma once

#include "../../layers/causal_lm_templates/text_decoder_layer.hpp"
#include "../../layers/common_modules.hpp"
#include "qwen3_vl_attention.hpp"

#include <infinicore/nn/embedding.hpp>
#include <infinicore/nn/rmsnorm.hpp>
#include <infinicore/tensor.hpp>

#include <memory>
#include <vector>

namespace infinilm::models::qwen3_vl {

using Qwen3VLMLP = infinilm::layers::MLP;
using Qwen3VLDecoderLayer = infinilm::layers::causal_lm_templates::TextDecoderLayer<Qwen3VLAttention,
                                                                                    Qwen3VLMLP>;

struct Qwen3VLDeepStackEmbedding {
    infinicore::Tensor features;
    size_t token_offset;
    size_t decoder_layer;
};

class Qwen3VLTextModel : public infinicore::nn::Module {
public:
    Qwen3VLTextModel(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                     const infinicore::Device &device);

    infinicore::Tensor forward(const infinilm::InfinilmModel::Input &input) const;

    infinicore::Tensor forward_embeds(const infinicore::Tensor &inputs_embeds,
                                      const infinicore::Tensor &position_ids,
                                      const std::vector<Qwen3VLDeepStackEmbedding>
                                          &deepstack_embeddings
                                      = {}) const;

    infinicore::Tensor embed_tokens(const infinicore::Tensor &input_ids) const;

private:
    infinicore::Tensor run_layers(
        infinicore::Tensor hidden_states, const infinicore::Tensor &position_ids,
        const std::vector<Qwen3VLDeepStackEmbedding> &deepstack_embeddings) const;

    INFINICORE_NN_MODULE(infinicore::nn::Embedding, embed_tokens);
    INFINICORE_NN_MODULE_VEC(Qwen3VLDecoderLayer, layers);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm);

    bool skip_final_norm_{false};
};

} // namespace infinilm::models::qwen3_vl
