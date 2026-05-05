#pragma once

#include "minicpm_sala_decoder_layer.hpp"

#include "../../config/model_config.hpp"
#include "infinicore/nn/embedding.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/tensor.hpp"

#include <memory>
#include <string>
#include <vector>

namespace infinilm::models::minicpm_sala {

class MiniCPMSALAModel : public infinicore::nn::Module {
public:
    MiniCPMSALAModel(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                     const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &input_ids,
                               const infinicore::Tensor &position_ids) const;

    void reset_state();

    size_t hidden_size() const { return hidden_size_; }

protected:
    INFINICORE_NN_MODULE(infinicore::nn::Embedding, embed_tokens);
    INFINICORE_NN_MODULE_VEC(MiniCPMSALADecoderLayer, layers);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm);

private:
    size_t hidden_size_;
};

} // namespace infinilm::models::minicpm_sala

