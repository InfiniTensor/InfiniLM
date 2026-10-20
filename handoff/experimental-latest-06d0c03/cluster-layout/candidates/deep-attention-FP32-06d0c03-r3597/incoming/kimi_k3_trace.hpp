#pragma once
#include "runtime/runtime.hpp"
#include "global_state/forward_context.hpp"
namespace infinilm::models::kimi_k3 {
void trace_hidden(runtime::Runtime&,const runtime::Tensor&,const global_state::ForwardContext&,size_t,const char*);
}
