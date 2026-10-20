#pragma once
#include "runtime/runtime.hpp"
#include "layers/rotary_embedding/rope.hpp"
#include <utility>
namespace infinilm::models::glm_moe_dsa {
// The reference indexer rotates adjacent input pairs, then stores all rotated
// even channels followed by all rotated odd channels. The pass-through tail
// stays in its original order. This differs from ordinary GPT-J output layout.
std::pair<runtime::Tensor,runtime::Tensor> indexer_rotary(
 runtime::Runtime &runtime,const layers::RoPE &rope,
 const runtime::Tensor &query,const runtime::Tensor &normalized_key,
 const runtime::Tensor &positions,size_t rotary_dim);
}
