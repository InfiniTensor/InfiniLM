#ifndef INFINI_OPS_CUDA_SILU_AND_MUL_INFINILM_KERNEL_H_
#define INFINI_OPS_CUDA_SILU_AND_MUL_INFINILM_KERNEL_H_

#include "base/silu_and_mul_infinilm.h"
#include "native/cuda/ops/silu_and_mul/kernel.h"

namespace infini::ops {

template <typename Backend>
class CudaSiluAndMulInfinilm : public SiluAndMulInfinilm {
 public:
  CudaSiluAndMulInfinilm(const Tensor input, Tensor out)
      : SiluAndMulInfinilm{input, out}, silu_and_mul_{input, out} {}

  void operator()(const Tensor input, Tensor out) const override {
    silu_and_mul_.set_stream(stream_);
    silu_and_mul_(input, out);
  }

 private:
  mutable CudaSiluAndMul<Backend> silu_and_mul_;
};

}  // namespace infini::ops

#endif
