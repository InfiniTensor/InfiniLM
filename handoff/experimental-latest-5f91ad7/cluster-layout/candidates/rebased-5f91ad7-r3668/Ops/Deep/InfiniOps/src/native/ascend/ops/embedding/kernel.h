#ifndef INFINI_OPS_ASCEND_EMBEDDING_KERNEL_H_
#define INFINI_OPS_ASCEND_EMBEDDING_KERNEL_H_

#include <algorithm>
#include <cassert>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_embedding.h"
#include "aclnnop/aclnn_embedding_renorm.h"
#include "base/embedding.h"
#include "native/ascend/common.h"
#include "native/ascend/workspace_pool_.h"
#include "operator.h"

namespace infini::ops {

template <>
class Operator<Embedding, Device::Type::kAscend> : public Embedding {
 public:
  Operator(const Tensor input, const Tensor weight,
           const std::optional<int64_t> padding_idx,
           const std::optional<double> max_norm, const double norm_type,
           const bool scale_grad_by_freq, const bool sparse, Tensor out)
      : Embedding(input, weight, padding_idx, max_norm, norm_type,
                  scale_grad_by_freq, sparse, out),
        input_cache_(input),
        weight_cache_(weight),
        out_cache_(out) {
    assert((weight_dtype_ == DataType::kFloat16 ||
            weight_dtype_ == DataType::kBFloat16 ||
            weight_dtype_ == DataType::kFloat32) &&
           "`Embedding`: Ascend path supports `float16`, `bfloat16`, and "
           "`float32` weights");
  }

  Operator(const Tensor input, const Tensor weight, Tensor out)
      : Operator(input, weight, std::nullopt, std::nullopt, 2.0, false, false,
                 out) {}

  /// \deprecated Use the overload that also accepts `max_norm` and
  /// `norm_type` instead.
  [[deprecated("Use the PyTorch-compatible overload instead.")]]
  Operator(const Tensor input, const Tensor weight, const int64_t padding_idx,
           const bool scale_grad_by_freq, const bool sparse, Tensor out)
      : Operator(input, weight, padding_idx, std::nullopt, 2.0,
                 scale_grad_by_freq, sparse, out) {}

  ~Operator() {
    if (!ascend::IsAclRuntimeAlive()) return;

    input_cache_.release();
    weight_cache_.release();
    out_cache_.release();
  }

  void operator()(const Tensor input, const Tensor weight,
                  const std::optional<int64_t> /*padding_idx*/,
                  const std::optional<double> max_norm, const double norm_type,
                  const bool /*scale_grad_by_freq*/, const bool /*sparse*/,
                  Tensor out) const override {
    auto stream = static_cast<aclrtStream>(stream_);

    auto t_weight = weight_cache_.get(const_cast<void*>(weight.data()));
    auto t_input = input_cache_.get(const_cast<void*>(input.data()));
    auto t_out = out_cache_.get(out.data());

    if (max_norm.has_value() && !renorm_executor_) {
      auto ret = aclnnEmbeddingRenormGetWorkspaceSize(
          t_weight, t_input, *max_norm, norm_type, &renorm_ws_size_,
          &renorm_executor_);
      assert(ret == ACL_SUCCESS &&
             "`aclnnEmbeddingRenormGetWorkspaceSize` failed");
      aclSetAclOpExecutorRepeatable(renorm_executor_);
    } else if (max_norm.has_value()) {
      aclSetInputTensorAddr(renorm_executor_, 0, t_weight,
                            const_cast<void*>(weight.data()));
      aclSetInputTensorAddr(renorm_executor_, 1, t_input,
                            const_cast<void*>(input.data()));
    }

    if (!embedding_executor_) {
      auto ret = aclnnEmbeddingGetWorkspaceSize(
          t_weight, t_input, t_out, &embedding_ws_size_, &embedding_executor_);
      assert(ret == ACL_SUCCESS && "`aclnnEmbeddingGetWorkspaceSize` failed");
      aclSetAclOpExecutorRepeatable(embedding_executor_);
    } else {
      aclSetInputTensorAddr(embedding_executor_, 0, t_weight,
                            const_cast<void*>(weight.data()));
      aclSetInputTensorAddr(embedding_executor_, 1, t_input,
                            const_cast<void*>(input.data()));
      aclSetOutputTensorAddr(embedding_executor_, 0, t_out, out.data());
    }

    const auto workspace_size = std::max(renorm_ws_size_, embedding_ws_size_);
    auto& arena = ascend::GetWorkspacePool().Ensure(stream, workspace_size);

    if (max_norm.has_value()) {
      auto ret = aclnnEmbeddingRenorm(arena.buf, renorm_ws_size_,
                                      renorm_executor_, stream);
      assert(ret == ACL_SUCCESS && "`aclnnEmbeddingRenorm` failed");
    }

    auto ret = aclnnEmbedding(arena.buf, embedding_ws_size_,
                              embedding_executor_, stream);
    assert(ret == ACL_SUCCESS && "`aclnnEmbedding` failed");
  }

 private:
  mutable ascend::AclTensorCache input_cache_;

  mutable ascend::AclTensorCache weight_cache_;

  mutable ascend::AclTensorCache out_cache_;

  mutable aclOpExecutor* renorm_executor_ = nullptr;

  mutable aclOpExecutor* embedding_executor_ = nullptr;

  mutable uint64_t renorm_ws_size_ = 0;

  mutable uint64_t embedding_ws_size_ = 0;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_ASCEND_EMBEDDING_KERNEL_H_
