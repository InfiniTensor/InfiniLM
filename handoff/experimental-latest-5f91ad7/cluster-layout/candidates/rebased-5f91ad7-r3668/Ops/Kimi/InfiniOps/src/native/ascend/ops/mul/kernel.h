#ifndef INFINI_OPS_ASCEND_MUL_KERNEL_H_
#define INFINI_OPS_ASCEND_MUL_KERNEL_H_

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_mul.h"
#include "base/mul.h"
#include "native/ascend/common.h"
#include "native/ascend/workspace_pool_.h"
#include "operator.h"

namespace infini::ops {

template <>
class Operator<Mul, Device::Type::kAscend> : public Mul {
 public:
  Operator(const Tensor input, const Tensor other, Tensor out)
      : Mul(input, other, out),
        in_cache_(input),
        oth_cache_(other),
        out_cache_(out) {}

  ~Operator() {
    if (!ascend::IsAclRuntimeAlive()) return;

    // Null cached descriptors — see `AclTensorCache::release()`.
    in_cache_.release();
    oth_cache_.release();
    out_cache_.release();
  }

  void operator()(const Tensor input, const Tensor other,
                  Tensor out) const override {
    auto stream = static_cast<aclrtStream>(stream_);
    auto t_in = in_cache_.get(const_cast<void*>(input.data()));
    auto t_oth = oth_cache_.get(const_cast<void*>(other.data()));
    auto t_out = out_cache_.get(out.data());

    if (!executor_) {
      aclnnMulGetWorkspaceSize(t_in, t_oth, t_out, &ws_size_, &executor_);
      aclSetAclOpExecutorRepeatable(executor_);
    } else {
      aclSetInputTensorAddr(executor_, 0, t_in,
                            const_cast<void*>(input.data()));
      aclSetInputTensorAddr(executor_, 1, t_oth,
                            const_cast<void*>(other.data()));
      aclSetOutputTensorAddr(executor_, 0, t_out, out.data());
    }

    auto& arena = ascend::GetWorkspacePool().Ensure(stream, ws_size_);
    aclnnMul(arena.buf, ws_size_, executor_, stream);
  }

 private:
  mutable ascend::AclTensorCache in_cache_;

  mutable ascend::AclTensorCache oth_cache_;

  mutable ascend::AclTensorCache out_cache_;

  mutable aclOpExecutor* executor_ = nullptr;

  mutable uint64_t ws_size_ = 0;
};

}  // namespace infini::ops

#endif
