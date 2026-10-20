#ifndef INFINI_OPS_LINKED_TORCH_OPS_RMS_NORM_H_
#define INFINI_OPS_LINKED_TORCH_OPS_RMS_NORM_H_

#include "base/rms_norm.h"
#include "torch/tensor_.h"

namespace infini::ops::linked::torch {

template <typename Backend>
class TorchRmsNorm : public ::infini::ops::RmsNorm {
 public:
  TorchRmsNorm(const Tensor input, const Tensor weight, float eps, Tensor out)
      : ::infini::ops::RmsNorm{input, weight, eps, out},
        weight_shape_{weight.shape()},
        weight_strides_{weight.strides()},
        input_type_{input.dtype()},
        weight_type_{weight.dtype()},
        out_type_{out.dtype()},
        is_input_contiguous_{input.IsContiguous()},
        is_weight_contiguous_{weight.IsContiguous()},
        is_out_contiguous_{out.IsContiguous()},
        device_index_{out.device().index()} {}

  TorchRmsNorm(const Tensor input, const Tensor weight, Tensor out)
      : TorchRmsNorm{input, weight, 1e-6f, out} {}

  using ::infini::ops::RmsNorm::operator();

  void operator()(const Tensor input, const Tensor weight, float eps,
                  Tensor out) const override {
    const typename Backend::StreamGuard stream_guard{
        Backend::GetStreamFromExternal(stream_, device_index_)};
    auto at_input = ToAtenTensor<Backend::kDeviceType>(
        const_cast<void*>(input.data()), input_shape_, input_strides_,
        input_type_, device_index_);
    auto at_weight = ToAtenTensor<Backend::kDeviceType>(
        const_cast<void*>(weight.data()), weight_shape_, weight_strides_,
        weight_type_, device_index_);
    auto at_out = ToAtenTensor<Backend::kDeviceType>(
        out.data(), out_shape_, out_strides_, out_type_, device_index_);

    auto backend_input =
        is_input_contiguous_ ? at_input : at_input.contiguous();
    auto backend_weight =
        is_weight_contiguous_ ? at_weight : at_weight.contiguous();
    auto backend_out = is_out_contiguous_
                           ? at_out
                           : at::empty(at_out.sizes(), at_out.options());

    Backend::Call(backend_out, backend_input, backend_weight,
                  static_cast<double>(eps));
    if (!is_out_contiguous_) {
      at_out.copy_(backend_out);
    }
  }

 private:
  Tensor::Shape weight_shape_;

  Tensor::Strides weight_strides_;

  DataType input_type_;

  DataType weight_type_;

  DataType out_type_;

  bool is_input_contiguous_{false};

  bool is_weight_contiguous_{false};

  bool is_out_contiguous_{false};

  int device_index_{0};
};

}  // namespace infini::ops::linked::torch

#endif  // INFINI_OPS_LINKED_TORCH_OPS_RMS_NORM_H_
