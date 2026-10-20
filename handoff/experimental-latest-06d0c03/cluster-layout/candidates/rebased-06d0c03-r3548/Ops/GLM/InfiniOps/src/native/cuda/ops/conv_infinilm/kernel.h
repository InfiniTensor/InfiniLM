#ifndef INFINI_OPS_CUDA_CONV_INFINILM_KERNEL_H_
#define INFINI_OPS_CUDA_CONV_INFINILM_KERNEL_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "base/conv_infinilm.h"
#include "native/cuda/ops/convolution/kernel.h"

namespace infini::ops {

template <typename Backend>
class CudaConvInfinilm : public ConvInfinilm {
 public:
  CudaConvInfinilm(const Tensor input, const Tensor weight,
                   std::optional<Tensor> bias,
                   const std::vector<int64_t> padding,
                   const std::vector<int64_t> stride,
                   const std::vector<int64_t> dilation, const int64_t groups,
                   Tensor out)
      : ConvInfinilm{input,  weight,   bias,   padding,
                     stride, dilation, groups, out},
        output_padding_(input.ndim() - 2, 0),
        convolution_{input, weight,          bias,   stride, padding, dilation,
                     false, output_padding_, groups, out} {}

  void operator()(const Tensor input, const Tensor weight,
                  std::optional<Tensor> bias,
                  const std::vector<int64_t> padding,
                  const std::vector<int64_t> stride,
                  const std::vector<int64_t> dilation, const int64_t groups,
                  Tensor out) const override {
    convolution_.set_stream(stream_);
    convolution_(input, weight, bias, stride, padding, dilation, false,
                 output_padding_, groups, out);
  }

 private:
  std::vector<int64_t> output_padding_;

  mutable CudaConv<Backend, Convolution> convolution_;
};

}  // namespace infini::ops

#endif
