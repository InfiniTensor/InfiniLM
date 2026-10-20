#ifndef INFINI_OPS_ASCEND_ARGMAX_KERNEL_H_
#define INFINI_OPS_ASCEND_ARGMAX_KERNEL_H_

#include <algorithm>
#include <cassert>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_argmax.h"
#include "aclnnop/aclnn_cast.h"
#include "base/argmax.h"
#include "native/ascend/common.h"
#include "native/ascend/workspace_pool_.h"
#include "operator.h"

namespace infini::ops {

// Greedy sampling reduction for one contiguous vocabulary vector. CANN
// ArgMax does not accept BF16, so that dtype is promoted before reduction.
template <>
class Operator<Argmax, Device::Type::kAscend> : public Argmax {
 public:
  Operator(const Tensor input, const std::optional<int64_t> dim,
           const bool keepdim, Tensor out)
      : Argmax(input, dim, keepdim, out),
        input_cache_(input),
        out_cache_(out),
        use_cast_(input.dtype() == DataType::kBFloat16) {
    assert(input.ndim() == 1 && input.numel() > 0 && input.IsContiguous() &&
           !dim.has_value() && !keepdim && out.ndim() == 0 &&
           out.numel() == 1 && out.dtype() == DataType::kInt64 &&
           (input.dtype() == DataType::kFloat16 ||
            input.dtype() == DataType::kBFloat16 ||
            input.dtype() == DataType::kFloat32) &&
           "Ascend `Argmax` provider 0 supports contiguous 1D float logits, "
           "no dim, keepdim=false, and a scalar int64 output");

    if (use_cast_) {
      const auto bytes = input.numel() * sizeof(float);
      auto ret = aclrtMalloc(&cast_data_, bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
      assert(ret == ACL_SUCCESS &&
             "Ascend `Argmax` failed to allocate BF16 promotion buffer");
      cast_cache_ = ascend::AclTensorCache(
          {static_cast<int64_t>(input.numel())}, ACL_FLOAT, cast_data_);
    }
  }

  ~Operator() {
    if (!ascend::IsAclRuntimeAlive()) return;

    input_cache_.release();
    out_cache_.release();
    if (use_cast_) {
      cast_cache_.release();
      aclrtFree(cast_data_);
    }
  }

  void operator()(const Tensor input, const std::optional<int64_t> dim,
                  const bool keepdim, Tensor out) const override {
    (void)dim;
    (void)keepdim;
    auto stream = static_cast<aclrtStream>(stream_);
    auto t_input = input_cache_.get(const_cast<void*>(input.data()));
    auto t_out = out_cache_.get(out.data());
    aclTensor* t_arg_input = t_input;

    if (use_cast_) {
      auto t_cast = cast_cache_.get(cast_data_);
      t_arg_input = t_cast;
      if (!cast_executor_) {
        auto ret = aclnnCastGetWorkspaceSize(t_input, ACL_FLOAT, t_cast,
                                             &cast_ws_size_, &cast_executor_);
        assert(ret == ACL_SUCCESS &&
               "Ascend `Argmax` BF16 cast workspace query failed");
        aclSetAclOpExecutorRepeatable(cast_executor_);
      } else {
        aclSetInputTensorAddr(cast_executor_, 0, t_input,
                              const_cast<void*>(input.data()));
        aclSetOutputTensorAddr(cast_executor_, 0, t_cast, cast_data_);
      }
    }

    if (!argmax_executor_) {
      auto ret = aclnnArgMaxGetWorkspaceSize(
          t_arg_input, 0, false, t_out, &argmax_ws_size_, &argmax_executor_);
      assert(ret == ACL_SUCCESS && "Ascend `Argmax` workspace query failed");
      aclSetAclOpExecutorRepeatable(argmax_executor_);
    } else {
      auto arg_input_data = use_cast_ ? cast_data_ : input.data();
      aclSetInputTensorAddr(argmax_executor_, 0, t_arg_input,
                            const_cast<void*>(arg_input_data));
      aclSetOutputTensorAddr(argmax_executor_, 0, t_out, out.data());
    }

    auto workspace_size = std::max(cast_ws_size_, argmax_ws_size_);
    auto& arena = ascend::GetWorkspacePool().Ensure(stream, workspace_size);
    if (use_cast_) {
      auto ret = aclnnCast(arena.buf, cast_ws_size_, cast_executor_, stream);
      assert(ret == ACL_SUCCESS && "Ascend `Argmax` BF16 cast failed");
    }
    auto ret =
        aclnnArgMax(arena.buf, argmax_ws_size_, argmax_executor_, stream);
    assert(ret == ACL_SUCCESS && "Ascend `Argmax` execution failed");
  }

 private:
  mutable ascend::AclTensorCache input_cache_;
  mutable ascend::AclTensorCache out_cache_;
  mutable ascend::AclTensorCache cast_cache_;
  bool use_cast_{false};
  void* cast_data_{nullptr};
  mutable aclOpExecutor* cast_executor_{nullptr};
  mutable uint64_t cast_ws_size_{0};
  mutable aclOpExecutor* argmax_executor_{nullptr};
  mutable uint64_t argmax_ws_size_{0};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_ASCEND_ARGMAX_KERNEL_H_
