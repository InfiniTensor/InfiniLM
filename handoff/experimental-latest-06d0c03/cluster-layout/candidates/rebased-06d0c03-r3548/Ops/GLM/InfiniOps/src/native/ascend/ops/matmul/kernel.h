#ifndef INFINI_OPS_ASCEND_MATMUL_KERNEL_H_
#define INFINI_OPS_ASCEND_MATMUL_KERNEL_H_

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_matmul.h"
#include "base/matmul.h"
#include "native/ascend/common.h"
#include "native/ascend/workspace_pool_.h"
#include "operator.h"

namespace infini::ops {

template <>
class Operator<Matmul, Device::Type::kAscend> : public Matmul {
 public:
  Operator(const Tensor input, const Tensor other, Tensor out)
      : Matmul(input, other, out),
        input_cache_(input),
        other_cache_(other),
        out_cache_(out) {}

  ~Operator() {
    if (!ascend::IsAclRuntimeAlive()) return;

    // Null cached descriptors — see `AclTensorCache::release()`.
    input_cache_.release();
    other_cache_.release();
    out_cache_.release();
  }

  void operator()(const Tensor input, const Tensor other,
                  Tensor out) const override {
    auto stream = static_cast<aclrtStream>(stream_);
    auto t_input = input_cache_.get(const_cast<void*>(input.data()));
    auto t_other = other_cache_.get(const_cast<void*>(other.data()));
    auto t_out = out_cache_.get(out.data());

    if (!executor_) {
      int8_t cube_math_type = 1;
      aclnnMatmulGetWorkspaceSize(t_input, t_other, t_out, cube_math_type,
                                  &ws_size_, &executor_);
      aclSetAclOpExecutorRepeatable(executor_);
    } else {
      aclSetInputTensorAddr(executor_, 0, t_input,
                            const_cast<void*>(input.data()));
      aclSetInputTensorAddr(executor_, 1, t_other,
                            const_cast<void*>(other.data()));
      aclSetOutputTensorAddr(executor_, 0, t_out, out.data());
    }

    auto& arena = ascend::GetWorkspacePool().Ensure(stream, ws_size_);
    aclnnMatmul(arena.buf, ws_size_, executor_, stream);
  }

 private:
  mutable ascend::AclTensorCache input_cache_;

  mutable ascend::AclTensorCache other_cache_;

  mutable ascend::AclTensorCache out_cache_;

  mutable aclOpExecutor* executor_ = nullptr;

  mutable uint64_t ws_size_ = 0;
};

}  // namespace infini::ops

#endif
