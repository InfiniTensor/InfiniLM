#pragma once

#include "../infinilm_model.hpp"
#include "../llama/llama_for_causal_lm.hpp"
#include "minicpmv_config.hpp"
#include "resampler.hpp"
#include "siglip_vision.hpp"

#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"

#include <memory>

namespace infinilm::models::minicpmv {

class MiniCPMVModel : public InfinilmModel {
public:
    MiniCPMVModel(const MiniCPMVConfig &config,
                  const infinicore::Device &device,
                  engine::distributed::RankInfo rank_info = engine::distributed::RankInfo());

    Output forward(const Input &input) const override;

    void reset_cache(const cache::CacheConfig *cache_config) override;

    uint32_t compress_kv_cache_inplace(uint32_t seq_len,
                                       size_t batch_size,
                                       const cache::KVCompressionConfig &cfg) override;

private:
    infinicore::Tensor replace_embeddings(const infinicore::Tensor &inputs_embeds,
                                          const infinicore::Tensor &vision_hidden,
                                          const infinicore::Tensor &image_bound) const;

    MiniCPMVConfig config_;
    engine::distributed::RankInfo rank_info_;

    INFINICORE_NN_MODULE(llama::LlamaForCausalLM, llm);
    INFINICORE_NN_MODULE(SiglipVisionModel, vpm);
    INFINICORE_NN_MODULE(Resampler, resampler);
};

} // namespace infinilm::models::minicpmv
