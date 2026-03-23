#include "attention/attention.hpp"
#include "attention/backends/flash_attn.hpp"
#include "attention/backends/flashinfer_attn.hpp"
#include "attention/backends/paged_attn.hpp"
#include "attention/backends/static_attn.hpp"
#include "linear/fused_linear.hpp"
#include "mlp/mlp.hpp"
#include "mlp/moe_mlp.hpp"
#include "text_causal_lm.hpp"
#include "text_decoder_layer.hpp"
#include "text_model.hpp"
#include <spdlog/spdlog.h>

namespace infinilm::layers {

using MLP = infinilm::layers::mlp::MLP;
using MoeMLP = infinilm::layers::mlp::MoeMLP;
using Attention = infinilm::layers::attention::Attention;

using RowParallelLinear = infinicore::nn::RowParallelLinear;
using ColumnParallelLinear = infinicore::nn::ColumnParallelLinear;
using ReplicatedLinear = infinicore::nn::Linear;
using QKVParallelLinear = infinilm::layers::linear::QKVParallelLinear;
using GateUpParallelLinear = infinilm::layers::linear::GateUpParallelLinear;

namespace attention {

using AttentionLayer = infinilm::layers::attention::AttentionLayer;

} // namespace attention
} // namespace infinilm::layers
