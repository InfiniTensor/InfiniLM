#pragma once
#include "mlp/mlp.hpp"
#include "mlp/moe_mlp.hpp"

#include "attention/attention.hpp"
#include "causal_lm_templates/text_causal_lm.hpp"
#include "causal_lm_templates/text_decoder_layer.hpp"
#include "causal_lm_templates/text_model.hpp"
#include "linear/linear.hpp"

namespace infinilm::layers {

using MLP = infinilm::layers::mlp::MLP;
using MoeMLP = infinilm::layers::moe_mlp::MoeMLP;

namespace attention {

using AttentionLayer = infinilm::layers::attention::AttentionLayer;

} // namespace attention
} // namespace infinilm::layers
