#pragma once

#if __has_include("infinicore/ops/gptq_marlin_gemm.hpp") && __has_include("infiniop/ops/awq_marlin_repack.h") && __has_include("infiniop/ops/gptq_marlin_repack.h")
#define INFINILM_ENABLE_MARLIN 1
#else
#define INFINILM_ENABLE_MARLIN 0
#endif
