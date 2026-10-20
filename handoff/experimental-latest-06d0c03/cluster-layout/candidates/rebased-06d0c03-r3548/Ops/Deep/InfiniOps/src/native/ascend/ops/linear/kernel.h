#ifndef INFINI_OPS_ASCEND_LINEAR_KERNEL_H_
#define INFINI_OPS_ASCEND_LINEAR_KERNEL_H_

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_addmm.h"
#include "aclnnop/aclnn_matmul.h"
#include "base/linear.h"
#include "native/ascend/common.h"
#include "native/ascend/workspace_pool_.h"
#include "operator.h"

namespace infini::ops {

template <>
class Operator<Linear, Device::Type::kAscend> : public Linear {
 public:
  Operator(const Tensor input, const Tensor weight, std::optional<Tensor> bias,
           Tensor out)
      : Linear(input, weight, bias, out),
        input_cache_(
            {static_cast<int64_t>(rows_), static_cast<int64_t>(input.size(-1))},
            ascend::ToAclDtype(input.dtype()), nullptr),
        weight_cache_(weight, true),
        out_cache_(
            {static_cast<int64_t>(rows_), static_cast<int64_t>(weight.size(0))},
            ascend::ToAclDtype(out.dtype()), nullptr) {
    if (has_bias_) {
      bias_cache_ = ascend::AclTensorCache(*bias);
      alpha_scalar_ = aclCreateScalar(&alpha_storage_, ACL_FLOAT);
      beta_scalar_ = aclCreateScalar(&beta_storage_, ACL_FLOAT);
    }
  }

  ~Operator() {
    if (!ascend::IsAclRuntimeAlive()) return;

    // Null cached descriptors — see `AclTensorCache::release()`.
    bias_cache_.release();
    input_cache_.release();
    weight_cache_.release();
    out_cache_.release();

    if (alpha_scalar_) aclDestroyScalar(alpha_scalar_);
    if (beta_scalar_) aclDestroyScalar(beta_scalar_);
  }

  void operator()(const Tensor input, const Tensor weight,
                  std::optional<Tensor> bias, Tensor out) const override {
    auto stream = static_cast<aclrtStream>(stream_);
    auto t_input = input_cache_.get(const_cast<void*>(input.data()));
    auto t_weight = weight_cache_.get(const_cast<void*>(weight.data()));
    auto t_out = out_cache_.get(out.data());

    if (has_bias_) {
      auto t_bias = bias_cache_.get(const_cast<void*>(bias->data()));

      if (!executor_) {
        aclnnAddmmGetWorkspaceSize(t_bias, t_input, t_weight, beta_scalar_,
                                   alpha_scalar_, t_out, 0, &ws_size_,
                                   &executor_);
        aclSetAclOpExecutorRepeatable(executor_);
      } else {
        aclSetInputTensorAddr(executor_, 0, t_bias,
                              const_cast<void*>(bias->data()));
        aclSetInputTensorAddr(executor_, 1, t_input,
                              const_cast<void*>(input.data()));
        aclSetInputTensorAddr(executor_, 2, t_weight,
                              const_cast<void*>(weight.data()));
        aclSetOutputTensorAddr(executor_, 0, t_out, out.data());
      }

      auto& arena = ascend::GetWorkspacePool().Ensure(stream, ws_size_);

      aclnnAddmm(arena.buf, ws_size_, executor_, stream);
    } else {
      if (!executor_) {
        int8_t cube_math_type = 1;
        aclnnMatmulGetWorkspaceSize(t_input, t_weight, t_out, cube_math_type,
                                    &ws_size_, &executor_);
        aclSetAclOpExecutorRepeatable(executor_);
      } else {
        aclSetInputTensorAddr(executor_, 0, t_input,
                              const_cast<void*>(input.data()));
        aclSetInputTensorAddr(executor_, 1, t_weight,
                              const_cast<void*>(weight.data()));
        aclSetOutputTensorAddr(executor_, 0, t_out, out.data());
      }

      auto& arena = ascend::GetWorkspacePool().Ensure(stream, ws_size_);
      aclnnMatmul(arena.buf, ws_size_, executor_, stream);
    }
  }

 private:
  mutable ascend::AclTensorCache bias_cache_;

  mutable ascend::AclTensorCache input_cache_;

  mutable ascend::AclTensorCache weight_cache_;

  mutable ascend::AclTensorCache out_cache_;

  float alpha_storage_ = 1.0f;

  float beta_storage_ = 1.0f;

  aclScalar* alpha_scalar_ = nullptr;

  aclScalar* beta_scalar_ = nullptr;

  mutable aclOpExecutor* executor_ = nullptr;

  mutable uint64_t ws_size_ = 0;
};

}  // namespace infini::ops

#endif
