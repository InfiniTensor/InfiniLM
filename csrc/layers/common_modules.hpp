#pragma once
#include "mlp/mlp.hpp"
#include "mlp/moe_mlp.hpp"

#include "attention/attention.hpp"
#include "linear/linear.hpp"
#include "text_causal_lm.hpp"
#include "text_decoder_layer.hpp"
#include "text_model.hpp"

namespace infinilm::layers {
using MLP = infinilm::layers::mlp::MLP;
using MoeMLP = infinilm::layers::moe_mlp::MoeMLP;

namespace attention {
using Attention = infinilm::layers::attention::Attention;

using AttentionLayer = infinilm::layers::attention::AttentionLayer;

} // namespace attention
} // namespace infinilm::layers