#pragma once

#if __has_include("infinicore/ops/gptq_marlin_gemm.hpp")
#define INFINILM_ENABLE_MARLIN 0
#else
#define INFINILM_ENABLE_MARLIN 0
#endif
