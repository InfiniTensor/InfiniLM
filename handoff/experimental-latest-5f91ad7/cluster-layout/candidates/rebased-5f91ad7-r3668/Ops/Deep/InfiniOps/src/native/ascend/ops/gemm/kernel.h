#ifndef INFINI_OPS_ASCEND_GEMM_KERNEL_H_
#define INFINI_OPS_ASCEND_GEMM_KERNEL_H_

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_addmm.h"
#include "aclnnop/aclnn_baddbmm.h"
#include "base/gemm.h"
#include "native/ascend/common.h"
#include "native/ascend/workspace_pool_.h"
#include "operator.h"

namespace infini::ops {

template <>
class Operator<Gemm, Device::Type::kAscend> : public Gemm {
 public:
  Operator(const Tensor a, const Tensor b, const std::optional<Tensor> c,
           std::optional<float> alpha, std::optional<float> beta,
           std::optional<int> trans_a, std::optional<int> trans_b, Tensor y)
      : Gemm(a, b, c, alpha, beta, trans_a, trans_b, y),
        batched_{batch_count_ > 1},
        alpha_val_{alpha.value_or(1.0f)},
        beta_val_{0.0f},
        self_cache_(y),
        a_cache_(a, trans_a_),
        b_cache_(b, trans_b_),
        out_cache_(y) {
    alpha_scalar_ = aclCreateScalar(&alpha_val_, ACL_FLOAT);
    beta_scalar_ = aclCreateScalar(&beta_val_, ACL_FLOAT);
  }

  Operator(const Tensor a, const Tensor b, Tensor y)
      : Operator{a,
                 b,
                 std::nullopt,
                 std::nullopt,
                 std::nullopt,
                 std::nullopt,
                 std::nullopt,
                 y} {}

  using Gemm::operator();

  ~Operator() {
    if (!ascend::IsAclRuntimeAlive()) return;

    // Null cached descriptors — see `AclTensorCache::release()`.
    self_cache_.release();
    a_cache_.release();
    b_cache_.release();
    out_cache_.release();

    if (alpha_scalar_) aclDestroyScalar(alpha_scalar_);
    if (beta_scalar_) aclDestroyScalar(beta_scalar_);
  }

  void operator()(const Tensor a, const Tensor b, const std::optional<Tensor> c,
                  std::optional<float> alpha, std::optional<float> beta,
                  std::optional<int> trans_a, std::optional<int> trans_b,
                  Tensor y) const override {
    static_cast<void>(EffectiveBeta(c, beta));
    auto stream = static_cast<aclrtStream>(stream_);

    auto t_self = self_cache_.get(y.data());
    auto t_a = a_cache_.get(const_cast<void*>(a.data()));
    auto t_b = b_cache_.get(const_cast<void*>(b.data()));
    auto t_out = out_cache_.get(y.data());

    if (!executor_) {
      if (batched_) {
        aclnnBaddbmmGetWorkspaceSize(t_self, t_a, t_b, beta_scalar_,
                                     alpha_scalar_, t_out, 0, &ws_size_,
                                     &executor_);
      } else {
        aclnnAddmmGetWorkspaceSize(t_self, t_a, t_b, beta_scalar_,
                                   alpha_scalar_, t_out, 0, &ws_size_,
                                   &executor_);
      }
      aclSetAclOpExecutorRepeatable(executor_);
    } else {
      aclSetInputTensorAddr(executor_, 0, t_self, y.data());
      aclSetInputTensorAddr(executor_, 1, t_a, const_cast<void*>(a.data()));
      aclSetInputTensorAddr(executor_, 2, t_b, const_cast<void*>(b.data()));
      aclSetOutputTensorAddr(executor_, 0, t_out, y.data());
    }

    auto& arena = ascend::GetWorkspacePool().Ensure(stream, ws_size_);

    if (batched_) {
      aclnnBaddbmm(arena.buf, ws_size_, executor_, stream);
    } else {
      aclnnAddmm(arena.buf, ws_size_, executor_, stream);
    }
  }

 private:
  bool batched_;

  float alpha_val_;

  float beta_val_;

  aclScalar* alpha_scalar_ = nullptr;

  aclScalar* beta_scalar_ = nullptr;

  mutable ascend::AclTensorCache self_cache_;

  mutable ascend::AclTensorCache a_cache_;

  mutable ascend::AclTensorCache b_cache_;

  mutable ascend::AclTensorCache out_cache_;

  mutable aclOpExecutor* executor_ = nullptr;

  mutable uint64_t ws_size_ = 0;
};

}  // namespace infini::ops

#endif
