#ifndef INFINI_OPS_CUDA_REARRANGE_INFINILM_KERNEL_H_
#define INFINI_OPS_CUDA_REARRANGE_INFINILM_KERNEL_H_

#include "base/rearrange_infinilm.h"
#include "native/cuda/ops/copy/kernel.h"

namespace infini::ops {

template <typename Backend>
class CudaRearrangeInfinilm : public RearrangeInfinilm {
 public:
  CudaRearrangeInfinilm(const Tensor input, Tensor out)
      : RearrangeInfinilm{input, out}, copy_{input, false, out} {}

  void operator()(const Tensor input, Tensor out) const override {
    copy_.set_stream(stream_);
    copy_(input, false, out);
  }

 private:
  mutable CudaCopy<Backend> copy_;
};

}  // namespace infini::ops

#endif
