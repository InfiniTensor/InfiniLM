#ifndef INFINI_OPS_CUDA_GELUTANH_INFINILM_KERNEL_H_
#define INFINI_OPS_CUDA_GELUTANH_INFINILM_KERNEL_H_

#include "base/gelutanh_infinilm.h"
#include "native/cuda/ops/gelu/kernel.h"

namespace infini::ops {

template <typename Backend>
class CudaGelutanhInfinilm : public GelutanhInfinilm {
 public:
  CudaGelutanhInfinilm(const Tensor input, Tensor out)
      : GelutanhInfinilm{input, out}, gelu_{input, "tanh", out} {}

  void operator()(const Tensor input, Tensor out) const override {
    gelu_.set_stream(stream_);
    gelu_(input, "tanh", out);
  }

 private:
  mutable CudaGelu<Backend> gelu_;
};

}  // namespace infini::ops

#endif
