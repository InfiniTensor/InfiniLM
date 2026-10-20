#ifndef INFINI_OPS_CUDA_SIGMOID_INFINILM_KERNEL_H_
#define INFINI_OPS_CUDA_SIGMOID_INFINILM_KERNEL_H_

#include "base/sigmoid_infinilm.h"
#include "native/cuda/ops/sigmoid/kernel.h"

namespace infini::ops {

template <typename Backend>
class CudaSigmoidInfinilm : public SigmoidInfinilm {
 public:
  CudaSigmoidInfinilm(const Tensor input, Tensor out)
      : SigmoidInfinilm{input, out}, sigmoid_{input, out} {}

  void operator()(const Tensor input, Tensor out) const override {
    sigmoid_.set_stream(stream_);
    sigmoid_(input, out);
  }

 private:
  mutable CudaSigmoid<Backend> sigmoid_;
};

}  // namespace infini::ops

#endif
