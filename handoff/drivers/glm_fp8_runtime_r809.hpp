#pragma once
#include "glm_fp8_model_r790.hpp"
#include "glm_fp8_cache_r807.hpp"
extern "C" int modern_glm_fp8_runtime_clear_r809(infinilm::runtime::Runtime*);
extern "C" const char*modern_glm_fp8_runtime_error_r809();
extern "C" int modern_glm_fp8_runtime_stats_r809(int,uint64_t*,size_t);
namespace infinilm::models::glm_moe_dsa {
bool fp8_cached_r809(runtime::Runtime&,const runtime::Tensor&,const runtime::Tensor&,const runtime::Tensor&,const runtime::Tensor&,const runtime::Tensor&,size_t,size_t);
}
