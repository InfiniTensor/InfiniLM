#ifndef INFINI_OPS_ASCEND_FILL_KERNEL_H_
#define INFINI_OPS_ASCEND_FILL_KERNEL_H_

#include <cassert>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_fill_scalar.h"
#include "aclnn_fill_tensor.h"
#include "base/fill.h"
#include "native/ascend/common.h"
#include "native/ascend/workspace_pool_.h"
#include "operator.h"

namespace infini::ops {

template <>
class Operator<Fill, Device::Type::kAscend> : public Fill {
 public:
  Operator(const Tensor input, const double value, Tensor out)
      : Fill(input, value, out), out_cache_(out) {
    Initialize(input, out);

    // `aclCreateScalar` stores the address, so the backing value is a member.
    scalar_storage_ = value;
    scalar_ = aclCreateScalar(&scalar_storage_, ACL_DOUBLE);
    assert(scalar_ != nullptr && "`AscendFill` failed to create scalar.");
  }

  Operator(const Tensor input, const Tensor value, Tensor out)
      : Fill(input, value, out), value_cache_(value), out_cache_(out) {
    assert(value.numel() == 1 &&
           "`AscendFill` requires a scalar Tensor value.");
    assert(value.device() == out.device() &&
           "`AscendFill` requires Tensor value and output on the same "
           "device.");
    Initialize(input, out);
  }

  ~Operator() override {
    if (scalar_ && ascend::IsAclRuntimeAlive()) {
      aclDestroyScalar(scalar_);
    }
  }

  void operator()(const Tensor input, const double /*value*/,
                  Tensor out) const override {
    (void)input;
    if (out.numel() == 0) return;

    auto stream = static_cast<aclrtStream>(stream_);
    auto t_out = out_cache_.get(out.data());

    aclOpExecutor* executor = nullptr;
    uint64_t workspace_size = 0;
    auto ret = aclnnInplaceFillScalarGetWorkspaceSize(
        t_out, scalar_, &workspace_size, &executor);
    assert(ret == ACL_SUCCESS &&
           "`aclnnInplaceFillScalarGetWorkspaceSize` failed.");

    auto& arena = ascend::GetWorkspacePool().Ensure(stream, workspace_size);
    ret = aclnnInplaceFillScalar(arena.buf, workspace_size, executor, stream);
    assert(ret == ACL_SUCCESS && "`aclnnInplaceFillScalar` failed.");
  }

  void operator()(const Tensor input, const Tensor value,
                  Tensor out) const override {
    (void)input;
    if (out.numel() == 0) return;

    auto stream = static_cast<aclrtStream>(stream_);
    auto t_value = value_cache_.get(const_cast<void*>(value.data()));
    auto t_out = out_cache_.get(out.data());

    aclOpExecutor* executor = nullptr;
    uint64_t workspace_size = 0;
    auto ret = aclnnInplaceFillTensorGetWorkspaceSize(
        t_out, t_value, &workspace_size, &executor);
    assert(ret == ACL_SUCCESS &&
           "`aclnnInplaceFillTensorGetWorkspaceSize` failed.");

    auto& arena = ascend::GetWorkspacePool().Ensure(stream, workspace_size);
    ret = aclnnInplaceFillTensor(arena.buf, workspace_size, executor, stream);
    assert(ret == ACL_SUCCESS && "`aclnnInplaceFillTensor` failed.");
  }

 private:
  static void Initialize(const Tensor input, const Tensor out) {
    assert(input.shape() == out.shape() &&
           "`AscendFill` requires input and output to have the same shape.");
    assert(input.dtype() == out.dtype() &&
           "`AscendFill` requires input and output to have the same dtype.");
    assert(input.device() == out.device() &&
           "`AscendFill` requires input and output on the same device.");
    assert(!out.HasBroadcastDim() &&
           "`AscendFill` output must not have broadcast dimensions.");
    assert(out.ndim() <= 8 &&
           "`AscendFill` does not support outputs with more than 8 "
           "dimensions.");
  }

  mutable ascend::AclTensorCache value_cache_;

  mutable ascend::AclTensorCache out_cache_;

  double scalar_storage_ = 0.0;

  aclScalar* scalar_ = nullptr;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_ASCEND_FILL_KERNEL_H_
