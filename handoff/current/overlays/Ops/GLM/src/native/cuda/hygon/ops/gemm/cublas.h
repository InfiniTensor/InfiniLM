#ifndef INFINI_OPS_HYGON_GEMM_CUBLAS_H_
#define INFINI_OPS_HYGON_GEMM_CUBLAS_H_

#include "native/cuda/hygon/blas.h"
#include "native/cuda/hygon/ops/gemm/handle_owner.h"
#include "native/cuda/ops/gemm/blas.h"

namespace infini::ops {

template <>
class Operator<Gemm, Device::Type::kHygon>
    : public BlasGemm<Blas<Device::Type::kHygon>> {
 public:
  using Parent=BlasGemm<Blas<Device::Type::kHygon>>;
  Operator(const Tensor a,const Tensor b,const std::optional<Tensor> c,
           std::optional<float> alpha,std::optional<float> beta,
           std::optional<int> trans_a,std::optional<int> trans_b,Tensor y)
      : Parent(a,b,c,alpha,beta,trans_a,trans_b,y),
        owner_(HygonGemmHandle::Acquire(y.device().index())) {}
  Operator(const Tensor a,const Tensor b,Tensor y)
      : Operator(a,b,std::nullopt,std::nullopt,std::nullopt,std::nullopt,std::nullopt,y) {}
 protected:
  cublasHandle_t GetHandle() const override { return owner_->get(); }
 private:
  std::shared_ptr<HygonGemmHandle> owner_;
};

}  // namespace infini::ops

#endif
