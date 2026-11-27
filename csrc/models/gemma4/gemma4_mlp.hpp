#pragma once

#include "../../config/model_config.hpp"
#include "../../engine/distributed/communication_group.hpp"
#include "../../layers/linear/fused_linear.hpp"
#include "../../layers/linear/linear.hpp"
#include "../../weights/weights.hpp"
#include "runtime/runtime.hpp"
#include "runtime/tensor.hpp"
#include <cstddef>
#include <memory>

namespace infinilm::models::gemma4 {

// Gemma 4 GeGLU MLP: down_proj(gelu_pytorch_tanh(gate_proj(x)) * up_proj(x)).
// Supports layer-specific intermediate_size (for use_double_wide_mlp on KV-shared layers).
class Gemma4TextMLP : public infinilm::weights::ModelWeights {
public:
    Gemma4TextMLP(infinilm::runtime::Runtime &runtime,
                  std::shared_ptr<infinilm::config::ModelConfig> model_config,
                  size_t hidden_size,
                  size_t intermediate_size,
                  const engine::distributed::RankInfo &rank_info);

    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime,
                                      const infinilm::runtime::Tensor &hidden_states) const;

    std::vector<std::shared_ptr<infinilm::weights::ModelWeights>> loading_children() const override {
        auto result = ModelWeights::loading_children();
        if (gate_up_proj_) {
            result.push_back(gate_up_proj_);
        }
        return result;
    }

private:
    size_t hidden_size_{0};
    size_t intermediate_size_{0};
    std::shared_ptr<infinilm::layers::linear::GateUpParallelLinear> gate_up_proj_;
    std::shared_ptr<infinilm::layers::linear::RowParallelLinear> down_proj_;
};

} // namespace infinilm::models::gemma4
