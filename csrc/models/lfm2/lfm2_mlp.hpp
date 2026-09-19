#pragma once

#include "../../layers/mlp/mlp.hpp"
#include <infinicore/ops/mul.hpp>
#include <infinicore/ops/silu.hpp>

namespace infinilm::models::lfm2 {

class Lfm2MLP : public infinilm::layers::mlp::MLP {
public:
    using infinilm::layers::mlp::MLP::MLP;

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const {
        if (hidden_states->dtype() == infinicore::DataType::F32) {
            return infinilm::layers::mlp::MLP::forward(hidden_states);
        }
        auto input = hidden_states;
        auto [gate, up] = gate_up_proj_->forward_split(input);
        // Reference F.silu(gate) materializes a low-precision tensor before
        // multiplication by up. A fused SwiGLU has a different rounding boundary.
        auto activated = infinicore::op::silu(gate);
        auto intermediate = infinicore::op::mul(activated, up);
        return down_proj_->forward(intermediate);
    }
};

} // namespace infinilm::models::lfm2
