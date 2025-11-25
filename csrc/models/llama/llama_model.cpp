#include "llama_model.hpp"
#include "infinicore/nn/embedding.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/nn/rope.hpp"
#include "infinicore/ops.hpp"

namespace infinilm::models::llama {

LlamaModel::LlamaModel(const LlamaConfig &config, const infinicore::Device &device)
    : config_(config) {
    // Initialize token embeddings
    INFINICORE_NN_MODULE_INIT(embed_tokens, config.vocab_size, config.hidden_size,
                              std::nullopt, infinicore::DataType::F32, device);

    // Initialize decoder layers
    INFINICORE_NN_MODULE_VEC_INIT(layers, config.num_hidden_layers, LlamaDecoderLayer,
                                   config, device);

    // Initialize final layer normalization
    INFINICORE_NN_MODULE_INIT(norm, config.hidden_size, config.rms_norm_eps,
                              infinicore::DataType::F32, device);

    // Initialize Rotary Position Embeddings (shared across all layers)
    // Use GPT-J-style inverse frequencies (default) and GPT_NEOX rotation pairing
    INFINICORE_NN_MODULE_INIT(rotary_emb, config.head_dim, config.max_position_embeddings,
                              config.rope_theta, infinicore::nn::RoPE::Algo::GPT_NEOX,
                              infinicore::DataType::F32, device);
}

infinicore::Tensor LlamaModel::forward(const infinicore::Tensor &input_ids,
                                        const infinicore::Tensor &position_ids,
                                        std::vector<void *> *kv_caches,
                                        const HookRegistry *hook_registry) const {
    // 1. Embed tokens: input_ids -> [batch, seq_len, hidden_size]
    auto hidden_states = embed_tokens_->forward(input_ids);
    if (hook_registry && hook_registry->has_hooks()) {
        hook_registry->call_hook("embed_tokens", hidden_states, -1);
    }

    // 2. Process through all decoder layers
    for (size_t i = 0; i < layers_.size(); ++i) {
        void *kv_cache = (kv_caches && i < kv_caches->size()) ? (*kv_caches)[i] : nullptr;
        std::string layer_prefix = "layer" + std::to_string(i);
        hidden_states = layers_.at(i)->forward(hidden_states, position_ids, kv_cache, hook_registry, layer_prefix, static_cast<int>(i));
    }

    // 3. Apply final layer normalization
    if (hook_registry && hook_registry->has_hooks()) {
        hook_registry->call_hook("before_final_norm", hidden_states, -1);
    }
    hidden_states = norm_->forward(hidden_states);
    if (hook_registry && hook_registry->has_hooks()) {
        hook_registry->call_hook("final_norm", hidden_states, -1);
    }

    return hidden_states;
}

} // namespace infinilm::models::llama
