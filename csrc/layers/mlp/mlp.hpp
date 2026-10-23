#pragma once
#include <typeinfo>

#include "../../config/model_config.hpp"
#include "../../weights/weights.hpp"
#include "../linear/linear.hpp"

namespace infinilm::backends {
enum class AttentionBackend;
}
namespace infinilm::layers::moe {
struct EPConfig;
}

namespace infinilm::layers::mlp {

/**
 * @brief MLP (Feed-Forward Network) module.
 *
 * Implements the MLP block with:
 * - Gate projection
 * - Up projection
 * - Down projection
 * - SiLU activation function
 *
 * Formula: down_proj(SiLU(gate_proj(x)) * up_proj(x))
 */
class MLP : public infinilm::weights::ModelWeights {
public:
    // The dense feed-forward computation is independent for each token.
    static constexpr bool supports_chunked_prefill = true;

    MLP(infinilm::runtime::Runtime &runtime, std::shared_ptr<infinilm::config::ModelConfig> model_config,
        size_t layer_idx, const engine::distributed::RankInfo &rank_info,
        backends::AttentionBackend attention_backend, const layers::moe::EPConfig &ep_config);

    /**
     * @brief Forward pass: compute MLP output
     *
     * @param hidden_states Input tensor of shape [batch, seq_len, hidden_size]
     * @return Output tensor of shape [batch, seq_len, hidden_size]
     */
    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime,
                                      const infinilm::runtime::Tensor &hidden_states) const;

    std::vector<std::shared_ptr<infinilm::weights::ModelWeights>> loading_children() const override {
        auto result = ModelWeights::loading_children();
        result.push_back(gate_up_proj_);
        return result;
    }

    // Module information
    size_t hidden_size() const { return hidden_size_; }
    size_t intermediate_size() const { return intermediate_size_; }

protected:
    std::shared_ptr<infinilm::layers::linear::GateUpParallelLinear> gate_up_proj_;
    std::shared_ptr<infinilm::layers::linear::RowParallelLinear> down_proj_;

    size_t hidden_size_;
    size_t intermediate_size_;
    bool use_bias_;
};

} // namespace infinilm::layers::mlp
