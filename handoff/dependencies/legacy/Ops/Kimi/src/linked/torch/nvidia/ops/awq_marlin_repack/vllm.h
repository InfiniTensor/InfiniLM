#ifndef INFINI_OPS_LINKED_TORCH_NVIDIA_OPS_AWQ_MARLIN_REPACK_VLLM_H_
#define INFINI_OPS_LINKED_TORCH_NVIDIA_OPS_AWQ_MARLIN_REPACK_VLLM_H_

#include "linked/torch/ops/awq_marlin_repack.h"
#include "torch/nvidia/c10.h"

namespace infini::ops::linked::torch::nvidia {

struct VllmAwqMarlinRepack : C10<Device::Type::kNvidia> {
  static at::Tensor Call(at::Tensor b_q_weight, int64_t size_k, int64_t size_n,
                         int64_t num_bits, bool is_a_8bit);
};

}  // namespace infini::ops::linked::torch::nvidia

namespace infini::ops::linked::torch {

extern template class TorchAwqMarlinRepack<
    ::infini::ops::linked::torch::nvidia::VllmAwqMarlinRepack>;

}  // namespace infini::ops::linked::torch

namespace infini::ops {

template <>
class Operator<AwqMarlinRepack, Device::Type::kNvidia, 16>
    : public linked::torch::TorchAwqMarlinRepack<
          linked::torch::nvidia::VllmAwqMarlinRepack> {
 public:
  using linked::torch::TorchAwqMarlinRepack<
      linked::torch::nvidia::VllmAwqMarlinRepack>::TorchAwqMarlinRepack;

  using linked::torch::TorchAwqMarlinRepack<
      linked::torch::nvidia::VllmAwqMarlinRepack>::operator();
};

}  // namespace infini::ops

#endif  // INFINI_OPS_LINKED_TORCH_NVIDIA_OPS_AWQ_MARLIN_REPACK_VLLM_H_
