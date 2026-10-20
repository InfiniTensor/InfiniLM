#ifndef INFINI_OPS_LINKED_TORCH_OPS_SILU_AND_MUL_H_
#define INFINI_OPS_LINKED_TORCH_OPS_SILU_AND_MUL_H_

#include "base/silu_and_mul.h"
#include "torch/tensor_.h"

namespace infini::ops::linked::torch {

template <typename Backend>
class TorchSiluAndMul : public ::infini::ops::SiluAndMul {
 public:
  TorchSiluAndMul(const Tensor input, Tensor out)
      : ::infini::ops::SiluAndMul{input, out},
        device_index_{out.device().index()} {}

  using ::infini::ops::SiluAndMul::operator();

  void operator()(const Tensor input, Tensor out) const override {
    const typename Backend::StreamGuard stream_guard{
        Backend::GetStreamFromExternal(stream_, device_index_)};
    auto at_input = ToAtenTensor<Backend::kDeviceType>(
        const_cast<void*>(input.data()), input_shape_, input_strides_,
        input_type_, device_index_);
    auto at_out = ToAtenTensor<Backend::kDeviceType>(
        out.data(), out_shape_, out_strides_, out_type_, device_index_);

    auto backend_input =
        is_input_contiguous_ ? at_input : at_input.contiguous();
    auto backend_out = is_out_contiguous_
                           ? at_out
                           : at::empty(at_out.sizes(), at_out.options());

    Backend::Call(backend_out, backend_input);
    if (!is_out_contiguous_) {
      at_out.copy_(backend_out);
    }
  }

 private:
  int device_index_{0};
};

}  // namespace infini::ops::linked::torch

#endif  // INFINI_OPS_LINKED_TORCH_OPS_SILU_AND_MUL_H_
