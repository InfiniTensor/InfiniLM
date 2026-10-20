#ifndef INFINI_OPS_CUDA_GELU_INFINILM_KERNEL_H_
#define INFINI_OPS_CUDA_GELU_INFINILM_KERNEL_H_

#include <string>

#include "base/gelu_infinilm.h"
#include "native/cuda/ops/gelu/kernel.h"

namespace infini::ops {

template <typename Backend>
class CudaGeluInfinilm : public GeluInfinilm {
 public:
  CudaGeluInfinilm(const Tensor input, const std::string approximate,
                   Tensor out)
      : GeluInfinilm{input, approximate, out},
        gelu_{input, approximate.empty() ? "none" : approximate, out} {}

  void operator()(const Tensor input, const std::string approximate,
                  Tensor out) const override {
    const std::string canonical_approximate =
        approximate.empty() ? "none" : approximate;
    gelu_.set_stream(stream_);
    gelu_(input, canonical_approximate, out);
  }

 private:
  mutable CudaGelu<Backend> gelu_;
};

}  // namespace infini::ops

#endif
