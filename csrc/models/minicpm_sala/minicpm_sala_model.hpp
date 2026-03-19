#pragma once

#include "minicpm_sala_decoder_layer.hpp"

#include "../../backends/attention_backends.hpp"
#include "../../cache/kv_cache.hpp"
#include "../../config/model_config.hpp"
#include "../../engine/distributed/distributed.hpp"

#include "infinicore/nn/embedding.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/nn/rope.hpp"
#include "infinicore/tensor.hpp"

#include <memory>
#include <string>
#include <vector>

namespace infinilm::models::minicpm_sala {

class MiniCPMSALAModel : public infinicore::nn::Module {
public:
    MiniCPMSALAModel(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                     const infinicore::Device &device,
                     engine::distributed::RankInfo rank_info = engine::distributed::RankInfo(),
                     backends::AttentionBackend attention_backend = backends::AttentionBackend::Default);

    infinicore::Tensor forward(const infinicore::Tensor &input_ids,
                               const infinicore::Tensor &position_ids,
                               std::optional<infinicore::Tensor> past_sequence_lengths,
                               std::optional<infinicore::Tensor> total_sequence_lengths,
                               std::optional<infinicore::Tensor> input_offsets,
                               std::optional<infinicore::Tensor> cu_seqlens,
                               std::optional<infinicore::Tensor> block_tables,
                               std::optional<infinicore::Tensor> slot_mapping) const;

    void reset_cache(const cache::CacheConfig *cache_config);

    size_t hidden_size() const { return hidden_size_; }
    double dim_model_base() const { return dim_model_base_; }

protected:
    INFINICORE_NN_MODULE(infinicore::nn::Embedding, embed_tokens);
    INFINICORE_NN_MODULE_VEC(MiniCPMSALADecoderLayer, layers);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm);
    INFINICORE_NN_MODULE(infinicore::nn::RoPE, rotary_emb);

private:
    std::shared_ptr<infinilm::config::ModelConfig> model_config_;
    engine::distributed::RankInfo rank_info_;
    backends::AttentionBackend attention_backend_;
    // MiniCPM-SALA is hybrid: minicpm4 vs lightning layers can have different KV shapes.
    // Use two StaticKVCache instances to avoid per-layer padding/copies during long prefill.
    std::shared_ptr<cache::StaticKVCache> kv_cache_minicpm4_;
    std::shared_ptr<cache::StaticKVCache> kv_cache_lightning_;
    std::vector<std::string> mixer_types_;
    infinicore::Device compute_device_;

    size_t hidden_size_;
    double scale_emb_;
    double dim_model_base_;
};

} // namespace infinilm::models::minicpm_sala

