#ifndef INFINI_OPS_ASCEND_SOFTMAX_KERNEL_H_
#define INFINI_OPS_ASCEND_SOFTMAX_KERNEL_H_

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_softmax.h"
#include "base/softmax.h"
#include "data_type.h"
#include "native/ascend/common.h"
#include "native/ascend/workspace_pool_.h"
#include "operator.h"

namespace infini::ops {

template <>
class Operator<Softmax, Device::Type::kAscend> : public Softmax {
 public:
  Operator(const Tensor input, const int64_t dim,
           const std::optional<DataType> dtype, Tensor out)
      : Softmax(input, dim, dtype, out),
        in_cache_(ValidateTensor(input)),
        out_cache_(ValidateTensor(out)) {
    assert(input_shape_ == out_shape_ &&
           "`Softmax` input and output shapes must match");
    assert(dim_ >= 0 && dim_ < static_cast<int64_t>(ndim_) &&
           "`Softmax` dim out of range");
    assert(!dtype_.has_value() || dtype_.value() == out_type_);
    assert(input_type_ == out_type_ &&
           "`Softmax` Ascend provider requires matching input and output "
           "dtypes");
    assert(input.device() == out.device() &&
           "`Softmax` input and output devices must match");
    assert((input_type_ == DataType::kFloat16 ||
            input_type_ == DataType::kBFloat16 ||
            input_type_ == DataType::kFloat32) &&
           "`Softmax` Ascend provider supports float16, bfloat16, and float32");
    assert(!out.HasBroadcastDim() &&
           "`Softmax` output must not have broadcasted dimensions");
  }

  ~Operator() {
    if (!ascend::IsAclRuntimeAlive()) return;

    in_cache_.release();
    out_cache_.release();
  }

  void operator()(const Tensor input, const int64_t dim,
                  const std::optional<DataType> dtype,
                  Tensor out) const override {
    assert((dim < 0 ? dim + static_cast<int64_t>(input.ndim()) : dim) == dim_ &&
           "`Softmax` dim changed after descriptor creation");
    assert(dtype == dtype_ &&
           "`Softmax` dtype changed after descriptor creation");

    if (row_count_ == 0 || dim_size_ == 0) return;

    auto stream = static_cast<aclrtStream>(stream_);
    auto t_in = in_cache_.get(const_cast<void*>(input.data()));
    auto t_out = out_cache_.get(out.data());

    if (!executor_) {
      aclnnSoftmaxGetWorkspaceSize(t_in, dim_, t_out, &ws_size_, &executor_);
      aclSetAclOpExecutorRepeatable(executor_);
    } else {
      aclSetInputTensorAddr(executor_, 0, t_in,
                            const_cast<void*>(input.data()));
      aclSetOutputTensorAddr(executor_, 0, t_out, out.data());
    }

    auto& arena = ascend::GetWorkspacePool().Ensure(stream, ws_size_);
    aclnnSoftmax(arena.buf, ws_size_, executor_, stream);
  }

 private:
  static Tensor ValidateTensor(const Tensor tensor) {
    const auto dtype = tensor.dtype();
    assert((dtype == DataType::kFloat16 || dtype == DataType::kBFloat16 ||
            dtype == DataType::kFloat32) &&
           "`Softmax` Ascend provider supports float16, bfloat16, and float32");
    return tensor;
  }

  mutable ascend::AclTensorCache in_cache_;

  mutable ascend::AclTensorCache out_cache_;

  mutable aclOpExecutor* executor_ = nullptr;

  mutable uint64_t ws_size_ = 0;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_ASCEND_SOFTMAX_KERNEL_H_
