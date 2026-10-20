#ifndef INFINI_OPS_CUDA_ZEROS_INFINILM_KERNEL_H_
#define INFINI_OPS_CUDA_ZEROS_INFINILM_KERNEL_H_

#include "base/zeros_infinilm.h"
#include "native/cuda/ops/fill/kernel.h"

namespace infini::ops {

template <typename Backend>
class CudaZerosInfinilm : public ZerosInfinilm {
 public:
  CudaZerosInfinilm(const Tensor input, Tensor out)
      : ZerosInfinilm{input, out}, fill_{input, 0.0, out} {}

  void operator()(const Tensor input, Tensor out) const override {
    fill_.set_stream(stream_);
    fill_(input, 0.0, out);
  }

 private:
  mutable CudaFill<Backend> fill_;
};

}  // namespace infini::ops

#endif
