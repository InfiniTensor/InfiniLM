#pragma once

#include "ernie4_5_moe_vl_decoder_layer.hpp"

#include "../../config/model_config.hpp"
#include "../infinilm_model.hpp"
#include "infinicore/nn/embedding.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/tensor.hpp"

#include <memory>
#include <vector>

namespace infinilm::models::ernie4_5_moe_vl {

// Text backbone: token embedding + heterogeneous decoder layers + final RMSNorm.
// Cannot reuse layers::causal_lm_templates::TextModel because the decoder layer
// forward needs token_type_ids (for modality-specific MoE routing).
class Ernie4_5_VLMoeModel : public infinicore::nn::Module {
public:
    Ernie4_5_VLMoeModel(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                        const infinicore::Device &device);

    // Standard text-only entry: derives token_type_ids = all-text.
    infinicore::Tensor forward(const infinilm::InfinilmModel::Input &input) const;

    // Multimodal entry: caller supplies merged embeddings + 3D position_ids +
    // per-token modality ids (0=text, 1=vision).
    infinicore::Tensor forward_embeds(const infinicore::Tensor &inputs_embeds,
                                      const infinicore::Tensor &position_ids,
                                      const infinicore::Tensor &token_type_ids) const;

    infinicore::Tensor embed_tokens(const infinicore::Tensor &input_ids) const;

protected:
    INFINICORE_NN_MODULE(infinicore::nn::Embedding, embed_tokens);
    INFINICORE_NN_MODULE_VEC(Ernie4_5_VLMoeDecoderLayer, layers);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm);
};

} // namespace infinilm::models::ernie4_5_moe_vl
