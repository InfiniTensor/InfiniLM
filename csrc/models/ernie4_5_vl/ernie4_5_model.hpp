#pragma once

#include "../../layers/common_modules.hpp"
#include "../infinilm_model.hpp"
#include "ernie4_5_decoder_layer.hpp"
#include "ernie4_5_vision.hpp"
#include "infinicore/nn/embedding.hpp"
#include "infinicore/nn/rmsnorm.hpp"

#include <vector>

namespace infinilm::models::ernie4_5_vl {

class Ernie45Model : public infinicore::nn::Module {
public:
    Ernie45Model(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                 const infinicore::Device &device);

    infinicore::Tensor forward(const InfinilmModel::Input &input) const;
    infinicore::Tensor forward(const InfinilmModel::Input &input,
                               const Ernie45VisionModel *vision_model) const;
    infinicore::Tensor forward_embeds(infinicore::Tensor hidden_states,
                                      const infinicore::Tensor &positions) const;

    void reset_cache(const cache::CacheConfig *cache_config);

protected:
    INFINICORE_NN_MODULE(Ernie45ResamplerModel, resampler_model);
    INFINICORE_NN_MODULE(infinicore::nn::Embedding, embed_tokens);
    INFINICORE_NN_MODULE_VEC(Ernie45DecoderLayer, layers);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm);

private:
    void replace_embeddings(infinicore::Tensor inputs_embeds,
                            const infinicore::Tensor &vision_hidden,
                            const infinicore::Tensor &image_bound) const;
    void apply_image_embeddings(infinicore::Tensor inputs_embeds,
                                const InfinilmModel::Input &input,
                                const Ernie45VisionModel &vision_model) const;

    std::shared_ptr<infinilm::config::ModelConfig> model_config_;
    std::shared_ptr<const Ernie45MropeCache> mrope_cache_;
};

} // namespace infinilm::models::ernie4_5_vl
