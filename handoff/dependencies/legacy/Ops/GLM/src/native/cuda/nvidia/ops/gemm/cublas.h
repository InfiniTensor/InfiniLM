#ifndef INFINI_OPS_NVIDIA_GEMM_CUBLAS_H_
#define INFINI_OPS_NVIDIA_GEMM_CUBLAS_H_

#include "native/cuda/nvidia/blas.h"
#include "native/cuda/nvidia/ops/gemm/handle_owner.h"
#include "native/cuda/ops/gemm/blas.h"

namespace infini::ops {

template <>
class Operator<Gemm, Device::Type::kNvidia, 0>
    : public BlasGemm<Blas<Device::Type::kNvidia>> {
 public:
  Operator(const Tensor a, const Tensor b, const std::optional<Tensor> c,
           std::optional<float> alpha, std::optional<float> beta,
           std::optional<int> trans_a, std::optional<int> trans_b, Tensor y)
      : BlasGemm<Blas<Device::Type::kNvidia>>{
            a, b, c, alpha, beta, trans_a, trans_b, y},
        handle_{NvidiaGemmHandle::Acquire(y.device().index())} {}

  Operator(const Tensor a, const Tensor b, Tensor y)
      : Operator{a, b, std::nullopt, std::nullopt, std::nullopt,
                 std::nullopt, std::nullopt, y} {}

  bool capture_safe() const override { return true; }

 protected:
  cublasHandle_t GetHandle() const override { return handle_->get(); }

 private:
  std::shared_ptr<NvidiaGemmHandle> handle_;
};

}  // namespace infini::ops

#endif
