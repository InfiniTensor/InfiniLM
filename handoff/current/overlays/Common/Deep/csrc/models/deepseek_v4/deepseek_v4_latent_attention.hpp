#pragma once
#include "runtime/runtime.hpp"
namespace infinilm::models::deepseek_v4 {
// Eager attention core on already decoded/rotated shared KV vectors. Selected
// SWA and compressed entries are one softmax domain; sink has zero value.
// Caller owns cache packing, compressor, causality and output inverse RoPE.
runtime::Tensor latent_attention(runtime::Runtime &,const runtime::Tensor &query,
 const runtime::Tensor &swa,const runtime::Tensor &compressed,const runtime::Tensor &sink,
 const runtime::Tensor &swa_indices,const runtime::Tensor &compressed_indices,
 const runtime::Tensor &lengths);
}
