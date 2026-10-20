#pragma once
#include "runtime/runtime.hpp"
namespace infinilm::models::deepseek_v4 {
void mapped_trace_r1052(runtime::Runtime&,const runtime::Tensor&,size_t layer,int point,size_t requests);
}
