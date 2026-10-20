#pragma once
#include "runtime/runtime.hpp"
namespace infinilm::models::glm_moe_dsa {
// Eager correctness baseline after KV expansion and rotary. Selection is shared
// across heads. Future entries are masked, and duplicate entries have set
// semantics, matching the reference scatter-mask path. No compressed cache owner.
runtime::Tensor sparse_attention(runtime::Runtime &,const runtime::Tensor &query,
 const runtime::Tensor &keys,const runtime::Tensor &values,const runtime::Tensor &indices,
 size_t past,float scale);
}
