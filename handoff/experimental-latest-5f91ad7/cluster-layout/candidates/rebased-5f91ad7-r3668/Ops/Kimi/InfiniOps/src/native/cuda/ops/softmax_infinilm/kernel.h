#ifndef INFINI_OPS_CUDA_SOFTMAX_INFINILM_KERNEL_H_
#define INFINI_OPS_CUDA_SOFTMAX_INFINILM_KERNEL_H_

#include "base/softmax_infinilm.h"
#include "native/cuda/ops/softmax/kernel.h"

namespace infini::ops {

template <typename Backend>
class CudaSoftmaxInfinilm : public SoftmaxInfinilm {
 public:
  CudaSoftmaxInfinilm(const Tensor input, const int64_t dim,
                      const std::optional<DataType> dtype, Tensor out)
      : SoftmaxInfinilm{input, dim, dtype, out},
        softmax_{input, dim, dtype, out} {}

  void operator()(const Tensor input, const int64_t dim,
                  const std::optional<DataType> dtype,
                  Tensor out) const override {
    softmax_.set_stream(stream_);
    softmax_(input, dim, dtype, out);
  }

 private:
  mutable CudaSoftmax<Backend> softmax_;
};

}  // namespace infini::ops

#endif
