#pragma once

#include "../../backends/attention_backends.hpp"
#include "../../config/model_config.hpp"
#include "../../engine/distributed/communication_group.hpp"
#include "../../global_state/forward_context.hpp"
#include "../../layers/embedding/embedding.hpp"
#include "../../layers/linear/linear.hpp"
#include "../../layers/moe/ep/ep_config.hpp"
#include "../../layers/normalization/rmsnorm.hpp"
#include "../../weights/weights.hpp"
#include "../infinilm_model.hpp"
#include "gemma4_audio.hpp"
#include "gemma4_config.hpp"
#include "gemma4_decoder_layer.hpp"
#include "gemma4_rope.hpp"
#include "gemma4_vision.hpp"
#include "runtime/runtime.hpp"
#include "runtime/tensor.hpp"
#include <memory>
#include <vector>

namespace infinilm::models::gemma4 {

class Gemma4TextModel : public infinilm::weights::ModelWeights {
public:
    Gemma4TextModel(infinilm::runtime::Runtime &runtime,
                    std::shared_ptr<infinilm::config::ModelConfig> model_config,
                    const engine::distributed::RankInfo &rank_info,
                    backends::AttentionBackend attention_backend,
                    const layers::moe::EPConfig &ep_config);

    infinilm::runtime::Tensor embed_tokens(infinilm::runtime::Runtime &runtime,
                                           const infinilm::runtime::Tensor &input_ids) const;

    infinilm::runtime::Tensor get_per_layer_inputs(infinilm::runtime::Runtime &runtime,
                                                   const infinilm::runtime::Tensor &input_ids) const;

    infinilm::runtime::Tensor project_per_layer_inputs(infinilm::runtime::Runtime &runtime,
                                                       const infinilm::runtime::Tensor &inputs_embeds,
                                                       const infinilm::runtime::Tensor &per_layer_inputs) const;

    infinilm::runtime::Tensor forward(const InfinilmModel::Input &input,
                                      global_state::ForwardContext &forward_context) const;

    infinilm::runtime::Tensor forward_embeds(const infinilm::runtime::Tensor &inputs_embeds,
                                             const infinilm::runtime::Tensor &per_layer_inputs,
                                             const infinilm::runtime::Tensor &positions,
                                             global_state::ForwardContext &forward_context,
                                             const Gemma4VisionSpans *vision_spans = nullptr) const;

    const Gemma4TextSpec &spec() const { return spec_; }

private:
    Gemma4TextSpec spec_;
    size_t local_layer_begin_{0};
    size_t local_layer_end_{0};
    size_t pp_size_{1};
    size_t pp_stage_{0};

    std::shared_ptr<infinilm::layers::Embedding> embed_tokens_;
    infinilm::runtime::Tensor embed_scale_tensor_; // [1] sqrt(hidden_size) in model dtype

    // Per-Layer Embedding (PLE) components (when hidden_size_per_layer_input > 0)
    std::shared_ptr<infinilm::layers::Embedding> embed_tokens_per_layer_;
    infinilm::runtime::Tensor ple_embed_scale_tensor_; // [1] sqrt(ple_dim) in model dtype
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> per_layer_model_projection_;
    infinilm::runtime::Tensor ple_proj_scale_tensor_;    // [1] hidden_size^-0.5
    infinilm::runtime::Tensor ple_combine_scale_tensor_; // [1] 2^-0.5
    std::shared_ptr<infinilm::layers::RMSNorm> per_layer_projection_norm_;

    std::shared_ptr<Gemma4RoPE> sliding_rope_;
    std::shared_ptr<Gemma4RoPE> full_rope_;

    std::vector<std::shared_ptr<Gemma4TextDecoderLayer>> layers_;
    std::shared_ptr<infinilm::layers::RMSNorm> norm_;
};

class Gemma4Model : public infinilm::weights::ModelWeights {
public:
    Gemma4Model(infinilm::runtime::Runtime &runtime,
                std::shared_ptr<infinilm::config::ModelConfig> model_config,
                const engine::distributed::RankInfo &rank_info,
                backends::AttentionBackend attention_backend,
                const layers::moe::EPConfig &ep_config);

    infinilm::runtime::Tensor forward(const InfinilmModel::Input &input,
                                      global_state::ForwardContext &forward_context) const;

    Gemma4TextModel &language_model() { return *language_model_; }

private:
    std::shared_ptr<infinilm::config::ModelConfig> model_config_;
    std::shared_ptr<Gemma4TextModel> language_model_;
    std::shared_ptr<Gemma4VisionModel> vision_tower_;
    std::shared_ptr<Gemma4MultimodalEmbedder> embed_vision_;
    std::shared_ptr<Gemma4AudioModel> audio_tower_;
    std::shared_ptr<Gemma4MultimodalEmbedder> embed_audio_;
};

template <typename ModelT>
class Gemma4CausalLMBase : public InfinilmModel {
public:
    Gemma4CausalLMBase(infinilm::runtime::Runtime &runtime,
                       std::shared_ptr<infinilm::config::ModelConfig> model_config,
                       const engine::distributed::RankInfo &rank_info,
                       backends::AttentionBackend attention_backend,
                       const layers::moe::EPConfig &ep_config);

    Output forward(const Input &input,
                   global_state::ForwardContext &forward_context) const override;

    void reset_cache(const cache::CacheConfig *cache_config,
                     global_state::ForwardContext &forward_context,
                     backends::AttentionBackend attention_backend,
                     const engine::distributed::RankInfo &rank_info) override;

protected:
    std::shared_ptr<ModelT> model_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> lm_head_;
    infinilm::runtime::Tensor last_token_shift_;
    infinilm::runtime::Tensor softcap_inv_half_;
    infinilm::runtime::Tensor softcap_two_;
    infinilm::runtime::Tensor softcap_neg_;
    double final_logit_softcapping_{30.0};
    size_t pp_size_{1};
    size_t pp_stage_{0};
};

using Gemma4ForConditionalGeneration = Gemma4CausalLMBase<Gemma4Model>;
using Gemma4ForCausalLM = Gemma4CausalLMBase<Gemma4TextModel>;

} // namespace infinilm::models::gemma4
