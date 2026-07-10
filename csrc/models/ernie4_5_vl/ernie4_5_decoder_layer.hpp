#pragma once

#include "../../layers/common_modules.hpp"
#include "ernie4_5_attention.hpp"
#include "ernie4_5_moe.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"

#include <memory>
#include <tuple>

namespace infinilm::models::ernie4_5_vl {

class Ernie45DecoderLayer : public infinicore::nn::Module {
public:
    Ernie45DecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                        size_t layer_idx,
                        std::shared_ptr<const Ernie45MropeCache> mrope_cache,
                        const infinicore::Device &device);

    std::tuple<infinicore::Tensor, infinicore::Tensor> forward(const infinicore::Tensor &positions,
                                                               infinicore::Tensor &hidden_states,
                                                               infinicore::Tensor &residual);

    infinicore::Tensor forward(const infinicore::Tensor &positions,
                               infinicore::Tensor &hidden_states);

protected:
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, input_layernorm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, post_attention_layernorm);
    INFINICORE_NN_MODULE(Ernie45Attention, self_attn);
    std::shared_ptr<infinicore::nn::Module> mlp_;

private:
    bool use_moe_{false};
};

} // namespace infinilm::models::ernie4_5_vl
