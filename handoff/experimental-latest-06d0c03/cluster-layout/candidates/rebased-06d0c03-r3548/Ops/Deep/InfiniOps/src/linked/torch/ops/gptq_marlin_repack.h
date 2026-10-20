#ifndef INFINI_OPS_LINKED_TORCH_OPS_GPTQ_MARLIN_REPACK_H_
#define INFINI_OPS_LINKED_TORCH_OPS_GPTQ_MARLIN_REPACK_H_

#include <utility>

#include "base/gptq_marlin_repack.h"
#include "torch/tensor_.h"

namespace infini::ops::linked::torch {

template <typename Backend>
class TorchGptqMarlinRepack : public ::infini::ops::GptqMarlinRepack {
 public:
  using ::infini::ops::GptqMarlinRepack::GptqMarlinRepack;
  using ::infini::ops::GptqMarlinRepack::operator();

  void operator()(const Tensor b_q_weight, const Tensor perm,
                  const int64_t size_k, const int64_t size_n,
                  const int64_t num_bits, const bool is_a_8bit,
                  Tensor out) const override {
    ValidateCallMetadata(b_q_weight, perm, size_k, size_n, num_bits, is_a_8bit,
                         out);

    const typename Backend::StreamGuard stream_guard{
        Backend::GetStreamFromExternal(stream_, device_index_)};
    auto at_b_q_weight = ToAtenTensor<Backend::kDeviceType>(
        const_cast<void*>(b_q_weight.data()), b_q_weight_metadata_.shape(),
        b_q_weight_metadata_.strides(), b_q_weight_metadata_.dtype(),
        device_index_);
    auto at_perm = ToAtenTensor<Backend::kDeviceType>(
        const_cast<void*>(perm.data()), perm_metadata_.shape(),
        perm_metadata_.strides(), perm_metadata_.dtype(), device_index_);
    auto at_out = ToAtenTensor<Backend::kDeviceType>(
        out.data(), out_metadata_.shape(), out_metadata_.strides(),
        out_metadata_.dtype(), device_index_);

    auto result = Backend::Call(std::move(at_b_q_weight), std::move(at_perm),
                                size_k, size_n, num_bits, is_a_8bit);
    at_out.copy_(result);
  }
};

}  // namespace infini::ops::linked::torch

#endif  // INFINI_OPS_LINKED_TORCH_OPS_GPTQ_MARLIN_REPACK_H_
