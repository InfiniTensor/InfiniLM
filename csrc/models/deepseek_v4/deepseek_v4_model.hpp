#pragma once

#include "../../config/model_config.hpp"
#include "deepseek_v4_decoder_layer.hpp"
#include "deepseek_v4_utils.hpp"
#include "infinicore/nn/embedding.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/tensor.hpp"

#include <memory>
#include <vector>

namespace infinilm::models::deepseek_v4 {

class DeepseekV4Model : public infinicore::nn::Module {
public:
    DeepseekV4Model(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                    const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &input_ids,
                               const infinicore::Tensor &positions) const;

private:
    void ensure_hc_head_fn_mat_right(const infinicore::Tensor &reference) const;
    infinicore::Tensor hc_head(const infinicore::Tensor &x) const;

    INFINICORE_NN_MODULE(infinicore::nn::Embedding, embed);
    INFINICORE_NN_MODULE_VEC(DeepseekV4DecoderLayer, layers);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm);
    INFINICORE_NN_PARAMETER(hc_head_base);
    INFINICORE_NN_PARAMETER(hc_head_fn);
    INFINICORE_NN_PARAMETER(hc_head_scale);

    size_t hidden_size_{0};
    size_t vocab_size_{0};
    size_t hc_mult_{0};
    double hc_eps_{0.0};

    mutable infinicore::Tensor hc_head_fn_mat_right_;
};

} // namespace infinilm::models::deepseek_v4
