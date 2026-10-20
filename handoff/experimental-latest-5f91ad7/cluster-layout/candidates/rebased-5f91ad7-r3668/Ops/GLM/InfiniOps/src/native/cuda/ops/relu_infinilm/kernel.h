#ifndef INFINI_OPS_CUDA_RELU_INFINILM_KERNEL_H_
#define INFINI_OPS_CUDA_RELU_INFINILM_KERNEL_H_

#include "base/relu_infinilm.h"
#include "native/cuda/ops/relu/kernel.h"

namespace infini::ops {

template <typename Backend>
class CudaReluInfinilm : public ReluInfinilm {
 public:
  CudaReluInfinilm(const Tensor input, Tensor out)
      : ReluInfinilm{input, out}, relu_{input, out} {}

  void operator()(const Tensor input, Tensor out) const override {
    relu_.set_stream(stream_);
    relu_(input, out);
  }

 private:
  mutable CudaRelu<Backend> relu_;
};

}  // namespace infini::ops

#endif
