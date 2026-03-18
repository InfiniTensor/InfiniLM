#include "TemplateCausalLM.hpp"
#include "TemplateDecoderLayer.hpp"
#include "TemplateModel.hpp"
#include "attention/flash_attn.hpp"
#include "attention/flashinfer_attn.hpp"
#include "attention/paged_attn.hpp"
#include "attention/static_attn.hpp"
#include "fused_linear.hpp"
#include "mlp/mlp.hpp"
#include "mlp/moe_mlp.hpp"

namespace infinilm::models::layers {

using StaticAttention = infinilm::models::layers::attention::StaticAttention;
using PagedAttention = infinilm::models::layers::attention::PagedAttention;
using FlashAttention = infinilm::models::layers::attention::FlashAttention;
using FlashInferAttention = infinilm::models::layers::attention::FlashInferAttention;

using MLP = infinilm::models::layers::mlp::MLP;
using MoeMLP = infinilm::models::layers::mlp::MoeMLP;

} // namespace infinilm::models::layers