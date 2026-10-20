#include "linked/torch/nvidia/ops/moe_wna16_marlin_gemm/vllm.h"

namespace infini::ops::linked::torch {

template class TorchMoeWna16MarlinGemm<
    ::infini::ops::linked::torch::nvidia::VllmMoeWna16MarlinGemm>;

}  // namespace infini::ops::linked::torch
