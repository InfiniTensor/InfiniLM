#ifndef INFINI_OPS_ASCEND_ADD_KERNEL_H_
#define INFINI_OPS_ASCEND_ADD_KERNEL_H_

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_add.h"
#include "base/add.h"
#include "data_type.h"
#include "native/ascend/common.h"
#include "native/ascend/workspace_pool_.h"
#include "operator.h"

namespace infini::ops {

template <>
class Operator<Add, Device::Type::kAscend> : public Add {
 public:
  Operator(const Tensor input, const Tensor other, const double alpha,
           Tensor out)
      : Add(input, other, alpha, out),
        in_cache_(input),
        oth_cache_(other),
        out_cache_(out) {
    // `aclCreateScalar` stores the pointer rather than copying the value, so
    // `alpha_storage_*` must remain alive for the lifetime of `alpha_`.
    // The alpha scalar type must match the tensor dtype: use int64 for integer
    // dtypes and float for floating-point dtypes.
    if (ascend::IsIntegerDtype(input.dtype())) {
      assert(alpha == static_cast<int64_t>(alpha) &&
             "operator `Add` requires integral `alpha` for integer tensors");
      alpha_int_storage_ = static_cast<int64_t>(alpha);
      alpha_ = aclCreateScalar(&alpha_int_storage_, ACL_INT64);
    } else {
      alpha_float_storage_ = static_cast<float>(alpha);
      alpha_ = aclCreateScalar(&alpha_float_storage_, ACL_FLOAT);
    }
  }

  Operator(const Tensor input, const Tensor other, Tensor out)
      : Operator(input, other, 1.0, out) {}

  ~Operator() {
    if (!ascend::IsAclRuntimeAlive()) return;

    // Null cached descriptors — see `AclTensorCache::release()`.  The
    // descriptors are still referenced by the Repeatable `executor_`, so
    // skipping `aclDestroyTensor` (and leaking the executor at shutdown)
    // avoids a double-free; see `64c367c`.
    in_cache_.release();
    oth_cache_.release();
    out_cache_.release();

    if (alpha_) aclDestroyScalar(alpha_);
  }

  void operator()(const Tensor input, const Tensor other,
                  const double /*alpha*/, Tensor out) const override {
    auto stream = static_cast<aclrtStream>(stream_);
    auto t_in = in_cache_.get(const_cast<void*>(input.data()));
    auto t_oth = oth_cache_.get(const_cast<void*>(other.data()));
    auto t_out = out_cache_.get(out.data());

    if (!executor_) {
      aclnnAddGetWorkspaceSize(t_in, t_oth, alpha_, t_out, &ws_size_,
                               &executor_);
      aclSetAclOpExecutorRepeatable(executor_);
    } else {
      aclSetInputTensorAddr(executor_, 0, t_in,
                            const_cast<void*>(input.data()));
      aclSetInputTensorAddr(executor_, 1, t_oth,
                            const_cast<void*>(other.data()));
      aclSetOutputTensorAddr(executor_, 0, t_out, out.data());
    }

    auto& arena = ascend::GetWorkspacePool().Ensure(stream, ws_size_);
    aclnnAdd(arena.buf, ws_size_, executor_, stream);
  }

 private:
  mutable ascend::AclTensorCache in_cache_;

  mutable ascend::AclTensorCache oth_cache_;

  mutable ascend::AclTensorCache out_cache_;

  mutable aclOpExecutor* executor_ = nullptr;

  mutable uint64_t ws_size_ = 0;

  // Stable address for `aclCreateScalar` (float).
  float alpha_float_storage_ = 1.0f;

  // Stable address for `aclCreateScalar` (int).
  int64_t alpha_int_storage_ = 1;

  aclScalar* alpha_ = nullptr;
};

}  // namespace infini::ops

#endif
