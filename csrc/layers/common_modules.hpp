#include "TemplateCausalLM.hpp"
#include "TemplateDecoderLayer.hpp"
#include "TemplateModel.hpp"
#include "attention/attention.hpp"
#include "attention/backends/flash_attn.hpp"
#include "attention/backends/flashinfer_attn.hpp"
#include "attention/backends/paged_attn.hpp"
#include "attention/backends/static_attn.hpp"
#include "fused_linear.hpp"
#include "mlp/mlp.hpp"
#include "mlp/moe_mlp.hpp"
#include <spdlog/spdlog.h>

namespace infinilm::layers {

using Attention = infinilm::layers::attention::Attention;

using MLP = infinilm::layers::mlp::MLP;
using MoeMLP = infinilm::layers::mlp::MoeMLP;

} // namespace infinilm::layers

namespace infinilm::layers::attention {

using AttentionLayer = infinilm::layers::attention::AttentionLayer;

} // namespace infinilm::layers::attention