#pragma once

#include "../common/moe_types.hpp"

#include "../../../config/model_config.hpp"
#include "../../../weights/weights.hpp"

#include <cstddef>
#include <memory>

namespace infinilm::engine::distributed {
struct RankInfo;
}

namespace infinilm::layers::moe {

struct EPConfig;

class FusedMoeExperts : public infinilm::weights::ModelWeights {
protected:
    infinilm::runtime::Device device_;

public:
    FusedMoeExperts(infinilm::runtime::Runtime &runtime,
                    std::shared_ptr<infinilm::config::ModelConfig> model_config,
                    const engine::distributed::RankInfo &rank_info,
                    const EPConfig &ep_config);

    const MoeWeights &moe_weights() const;

protected:
    infinilm::weights::Weight w13_weight_;
    infinilm::weights::Weight w2_weight_;

    size_t num_experts_{0};
    size_t hidden_size_{0};
    size_t intermediate_size_per_partition_{0};
    MoeWeights moe_weights_;
};

} // namespace infinilm::layers::moe
