#ifndef INFINI_OPS_ASCEND_COPY_KERNEL_H_
#define INFINI_OPS_ASCEND_COPY_KERNEL_H_

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_copy.h"
#include "base/copy.h"
#include "native/ascend/common.h"
#include "native/ascend/workspace_pool_.h"
#include "operator.h"

namespace infini::ops {

template <>
class Operator<Copy, Device::Type::kAscend> : public Copy {
 public:
  Operator(const Tensor src, const bool non_blocking, Tensor out)
      : Copy(src, non_blocking, out),
        in_cache_(BroadcastView(src, out)),
        out_cache_(out) {}

  void operator()(const Tensor src, const bool /*non_blocking*/,
                  Tensor out) const override {
    if (output_size_ == 0) return;

    auto stream = static_cast<aclrtStream>(stream_);
    auto t_in = in_cache_.get(const_cast<void*>(src.data()));
    auto t_out = out_cache_.get(out.data());

    // InplaceCopy executors are consumed by CANN even after marking them
    // repeatable. Reusing the pointer causes a double-free or use-after-free
    // on a later model layer, so acquire an executor for every invocation.
    aclOpExecutor* executor = nullptr;
    uint64_t workspace_size = 0;
    aclnnInplaceCopyGetWorkspaceSize(t_out, t_in, &workspace_size, &executor);

    auto& arena = ascend::GetWorkspacePool().Ensure(stream, workspace_size);
    aclnnInplaceCopy(arena.buf, workspace_size, executor, stream);
  }

 private:
  static Tensor BroadcastView(const Tensor src, const Tensor out) {
    return Tensor{const_cast<void*>(src.data()), out.shape(), src.dtype(),
                  src.device(), BroadcastStrides(src, out)};
  }

  // Descriptors must outlive the asynchronous ACLNN call. The cached
  // operator owns them, while each invocation still gets a fresh executor.
  mutable ascend::AclTensorCache in_cache_;

  mutable ascend::AclTensorCache out_cache_;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_ASCEND_COPY_KERNEL_H_
