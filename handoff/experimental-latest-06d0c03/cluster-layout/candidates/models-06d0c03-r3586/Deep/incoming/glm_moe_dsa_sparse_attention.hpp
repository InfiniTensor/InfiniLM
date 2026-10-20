#pragma once
#include "runtime/runtime.hpp"
namespace infinilm::models::glm_moe_dsa {
// Eager correctness baseline after KV expansion and rotary. Selection is shared
// across heads. Future entries are masked, and duplicate entries have set
// semantics, matching the reference scatter-mask path. No compressed cache owner.
runtime::Tensor sparse_attention(runtime::Runtime &,const runtime::Tensor &query,
 const runtime::Tensor &keys,const runtime::Tensor &values,const runtime::Tensor &indices,
 size_t past,float scale);
// Bank entries can be a selected subset. Causality follows original logical
// key positions rather than the compact bank's row numbers.
runtime::Tensor sparse_attention_mapped(runtime::Runtime &,const runtime::Tensor &query,
 const runtime::Tensor &keys,const runtime::Tensor &values,const runtime::Tensor &indices,
 const runtime::Tensor &logical_key_positions,size_t past,float scale);
runtime::Tensor sparse_attention_bounded(runtime::Runtime&,const runtime::Tensor&,const runtime::Tensor&,const runtime::Tensor&,const runtime::Tensor&,size_t,float);
runtime::Tensor sparse_attention_mapped_bounded(runtime::Runtime&,const runtime::Tensor&,const runtime::Tensor&,const runtime::Tensor&,const runtime::Tensor&,const runtime::Tensor&,size_t,float);
}
