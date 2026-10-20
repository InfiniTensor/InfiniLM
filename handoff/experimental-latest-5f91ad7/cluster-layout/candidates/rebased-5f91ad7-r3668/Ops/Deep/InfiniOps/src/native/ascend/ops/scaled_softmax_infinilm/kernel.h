#ifndef INFINI_OPS_ASCEND_SCALED_SOFTMAX_INFINILM_KERNEL_H_
#define INFINI_OPS_ASCEND_SCALED_SOFTMAX_INFINILM_KERNEL_H_

#include <cmath>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_mul.h"
#include "aclnn_softmax.h"
#include "base/scaled_softmax_infinilm.h"
#include "data_type.h"
#include "native/ascend/common.h"
#include "native/ascend/workspace_pool_.h"
#include "operator.h"

namespace infini::ops {

template <>
class Operator<ScaledSoftmaxInfinilm, Device::Type::kAscend>
    : public ScaledSoftmaxInfinilm {
 public:
  Operator(const Tensor input, double scale, Tensor out)
      : ScaledSoftmaxInfinilm(input, scale, out),
        in_cache_(input),
        out_cache_(out),
        temp_cache_(input),
        scale_storage_(static_cast<float>(scale)),
        needs_scale_(std::fabs(scale - 1.0) > 1e-6) {
    assert((dtype_ == DataType::kFloat16 || dtype_ == DataType::kBFloat16 ||
            dtype_ == DataType::kFloat32) &&
           "`ScaledSoftmaxInfinilm` Ascend path requires float16, bfloat16, or "
           "float32 input");
    assert(input.IsContiguous() &&
           "`ScaledSoftmaxInfinilm` Ascend path requires contiguous input");
    assert(out.IsContiguous() &&
           "`ScaledSoftmaxInfinilm` Ascend path requires contiguous output");

    temp_size_ = input.numel() * kDataTypeToSize.at(dtype_);
    scale_scalar_ = aclCreateScalar(&scale_storage_, ACL_FLOAT);
  }

  ~Operator() {
    if (!ascend::IsAclRuntimeAlive()) return;

    in_cache_.release();
    out_cache_.release();
    temp_cache_.release();

    if (scale_scalar_) aclDestroyScalar(scale_scalar_);
  }

  void operator()(const Tensor input, double scale, Tensor out) const override {
    assert(scale == scale_ &&
           "`ScaledSoftmaxInfinilm` scale changed after descriptor creation");

    auto stream = static_cast<aclrtStream>(stream_);
    auto t_in = in_cache_.get(const_cast<void*>(input.data()));
    auto t_out = out_cache_.get(out.data());
    aclTensor* t_softmax_in = t_in;
    void* softmax_in_data = const_cast<void*>(input.data());

    if (needs_scale_) {
      auto& temp =
          ascend::GetWorkspacePool().Ensure(stream, temp_size_, "temp");
      auto t_temp = temp_cache_.get(temp.buf);

      if (!muls_exec_) {
        aclnnMulsGetWorkspaceSize(t_in, scale_scalar_, t_temp, &muls_ws_,
                                  &muls_exec_);
        aclSetAclOpExecutorRepeatable(muls_exec_);
      } else {
        aclSetInputTensorAddr(muls_exec_, 0, t_in,
                              const_cast<void*>(input.data()));
        aclSetOutputTensorAddr(muls_exec_, 0, t_temp, temp.buf);
      }

      auto& muls_arena = ascend::GetWorkspacePool().Ensure(stream, muls_ws_);
      aclnnMuls(muls_arena.buf, muls_ws_, muls_exec_, stream);

      t_softmax_in = t_temp;
      softmax_in_data = temp.buf;
    }

    if (!softmax_exec_) {
      constexpr int64_t kLastDim = -1;
      aclnnSoftmaxGetWorkspaceSize(t_softmax_in, kLastDim, t_out, &softmax_ws_,
                                   &softmax_exec_);
      aclSetAclOpExecutorRepeatable(softmax_exec_);
    } else {
      aclSetInputTensorAddr(softmax_exec_, 0, t_softmax_in, softmax_in_data);
      aclSetOutputTensorAddr(softmax_exec_, 0, t_out, out.data());
    }

    auto& softmax_arena =
        ascend::GetWorkspacePool().Ensure(stream, softmax_ws_);
    aclnnSoftmax(softmax_arena.buf, softmax_ws_, softmax_exec_, stream);
  }

 private:
  mutable ascend::AclTensorCache in_cache_;

  mutable ascend::AclTensorCache out_cache_;

  mutable ascend::AclTensorCache temp_cache_;

  float scale_storage_{1.0f};

  aclScalar* scale_scalar_ = nullptr;

  bool needs_scale_{false};

  uint64_t temp_size_{0};

  mutable aclOpExecutor* muls_exec_ = nullptr;

  mutable uint64_t muls_ws_ = 0;

  mutable aclOpExecutor* softmax_exec_ = nullptr;

  mutable uint64_t softmax_ws_ = 0;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_ASCEND_SCALED_SOFTMAX_INFINILM_KERNEL_H_
