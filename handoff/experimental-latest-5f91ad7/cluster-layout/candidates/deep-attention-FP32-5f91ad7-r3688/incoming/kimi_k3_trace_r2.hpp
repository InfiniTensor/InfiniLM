#pragma once
#include "runtime/runtime.hpp"
#include "global_state/forward_context.hpp"
namespace infinilm::models::kimi_k3 {
bool trace_detail_layer(size_t);
void trace_hidden(runtime::Runtime&,const runtime::Tensor&,const global_state::ForwardContext&,size_t,const char*);
}
