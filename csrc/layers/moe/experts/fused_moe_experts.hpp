#pragma once

#include "../common/moe_types.hpp"

#include "../../../config/model_config.hpp"
#include "infinicore/nn/module.hpp"

#include <cstddef>
#include <memory>

namespace infinilm::layers::moe {

class FusedMoeExperts : public infinicore::nn::Module {
public:
    FusedMoeExperts(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                    const infinicore::Device &device);

    const MoeWeights &moe_weights() const;

    void process_weights_after_loading() override;

protected:
    INFINICORE_NN_PARAMETER(w13_weight);
    INFINICORE_NN_PARAMETER(w2_weight);
    INFINICORE_NN_PARAMETER(w13_weight_scale);
    INFINICORE_NN_PARAMETER(w2_weight_scale);

    size_t num_experts_{0};
    size_t hidden_size_{0};
    size_t intermediate_size_per_partition_{0};
    bool enable_hygon_w16a16_marlin_{false};
    bool enable_hygon_w8a8_marlin_{false};
    bool w16a16_marlin_packed_{false};
    bool w8a8_marlin_packed_{false};
    MoeWeights moe_weights_;
};

} // namespace infinilm::layers::moe
