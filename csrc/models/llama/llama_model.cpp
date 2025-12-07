#include "llama_model.hpp"
#include "infinicore/nn/embedding.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/nn/rope.hpp"
#include "infinicore/ops.hpp"

namespace infinilm::models::llama {

LlamaModel::LlamaModel(const LlamaConfig &config,
                       const infinicore::Device &device,
                       infinicore::DataType dtype,
                       engine::distributed::RankInfo rank_info)
    : config_(config) {
    // Initialize token embeddings
    INFINICORE_NN_MODULE_INIT(embed_tokens, config.vocab_size, config.hidden_size,
                              std::nullopt, dtype, device);

    // Initialize decoder layers
    INFINICORE_NN_MODULE_VEC_INIT(layers, config.num_hidden_layers, LlamaDecoderLayer,
                                  config, device, dtype, rank_info);

    // Initialize final layer normalization
    INFINICORE_NN_MODULE_INIT(norm, config.hidden_size, config.rms_norm_eps,
                              dtype, device);

    // Initialize Rotary Position Embeddings (shared across all layers)
    // Use GPT-J-style inverse frequencies (default) and GPT_NEOX rotation pairing
    INFINICORE_NN_MODULE_INIT(rotary_emb, config.head_dim, config.max_position_embeddings,
                              config.rope_theta, infinicore::nn::RoPE::Algo::GPT_NEOX,
                              dtype, device);

    for (auto &layer : layers_) {
        if (layer) {
            layer->set_rotary_emb(rotary_emb_);
        }
    }
}

infinicore::Tensor LlamaModel::forward(const infinicore::Tensor &input_ids,
                                       const infinicore::Tensor &position_ids,
                                       std::vector<void *> *kv_caches) const {
    // 1. Embed tokens: input_ids -> [batch, seq_len, hidden_size]
    auto hidden_states = embed_tokens_->forward(input_ids);

    // 2. Process through all decoder layers
    for (size_t i = 0; i < layers_.size(); ++i) {
        void *kv_cache = (kv_caches && i < kv_caches->size()) ? (*kv_caches)[i] : nullptr;
        hidden_states = layers_.at(i)->forward(hidden_states, position_ids, kv_cache);
    }

    // 3. Apply final layer normalization to last token only (aligns with transformers)

    // Narrow to last token: [batch, seq_len, hidden_size] -> [batch, 1, hidden_size]
    auto shape = hidden_states->shape();
    size_t seq_len = shape[1];
    auto last_token = hidden_states; //->narrow({{1, seq_len - 1, 1}});

    auto normalized_states = norm_->forward(hidden_states);
    auto normalized_last_token = normalized_states->narrow({{1, seq_len - 1, 1}});

    return normalized_last_token;
}

} // namespace infinilm::models::llama
