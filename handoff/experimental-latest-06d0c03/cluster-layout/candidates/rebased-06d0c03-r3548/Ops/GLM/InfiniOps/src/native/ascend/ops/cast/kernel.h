#ifndef INFINI_OPS_ASCEND_CAST_KERNEL_H_
#define INFINI_OPS_ASCEND_CAST_KERNEL_H_

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_cast.h"
#include "base/cast.h"
#include "native/ascend/common.h"
#include "native/ascend/workspace_pool_.h"
#include "operator.h"

namespace infini::ops {

template <>
class Operator<Cast, Device::Type::kAscend> : public Cast {
 public:
  Operator(const Tensor input, Tensor out)
      : Cast(input, out),
        in_cache_(input),
        out_cache_(out),
        acl_out_dtype_(ascend::ToAclDtype(out.dtype())) {}

  ~Operator() {
    if (!ascend::IsAclRuntimeAlive()) return;

    // Null cached descriptors — see `AclTensorCache::release()`.
    in_cache_.release();
    out_cache_.release();
  }

  void operator()(const Tensor input, Tensor out) const override {
    auto stream = static_cast<aclrtStream>(stream_);
    auto t_in = in_cache_.get(const_cast<void*>(input.data()));
    auto t_out = out_cache_.get(out.data());

    if (!executor_) {
      aclnnCastGetWorkspaceSize(t_in, acl_out_dtype_, t_out, &ws_size_,
                                &executor_);
      aclSetAclOpExecutorRepeatable(executor_);
    } else {
      aclSetInputTensorAddr(executor_, 0, t_in,
                            const_cast<void*>(input.data()));
      aclSetOutputTensorAddr(executor_, 0, t_out, out.data());
    }

    auto& arena = ascend::GetWorkspacePool().Ensure(stream, ws_size_);
    aclnnCast(arena.buf, ws_size_, executor_, stream);
  }

 private:
  mutable ascend::AclTensorCache in_cache_;

  mutable ascend::AclTensorCache out_cache_;

  aclDataType acl_out_dtype_;

  mutable aclOpExecutor* executor_ = nullptr;

  mutable uint64_t ws_size_ = 0;
};

}  // namespace infini::ops

#endif
