#ifndef INFINI_OPS_ASCEND_TOP_K_TOP_P_SAMPLING_FROM_LOGITS_KERNEL_H_
#define INFINI_OPS_ASCEND_TOP_K_TOP_P_SAMPLING_FROM_LOGITS_KERNEL_H_

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_cast.h"
#include "aclnnop/aclnn_top_k_top_p_sample.h"
#include "base/top_k_top_p_sampling_from_logits.h"
#include "data_type.h"
#include "native/ascend/common.h"
#include "native/ascend/workspace_pool_.h"
#include "operator.h"
#include "tensor.h"

namespace infini::ops {

template <>
class Operator<TopKTopPSamplingFromLogits, Device::Type::kAscend, 0>
    : public TopKTopPSamplingFromLogits {
 public:
  Operator(const Tensor logits, const Tensor top_k, const Tensor top_p,
           const std::optional<Tensor> indices,
           const std::string filter_apply_order, const bool deterministic,
           const bool check_nan, const std::optional<int64_t> seed,
           const std::optional<int64_t> offset, Tensor out)
      : TopKTopPSamplingFromLogits(logits, top_k, top_p, indices,
                                   filter_apply_order, deterministic, check_nan,
                                   seed, offset, out) {
    ValidateSupportedOptions(indices, filter_apply_order, deterministic,
                             check_nan);
    assert((dtype_ == DataType::kFloat16 || dtype_ == DataType::kBFloat16) &&
           "The Ascend `TopKTopPSamplingFromLogits` provider requires "
           "float16 or bfloat16 logits.");
    assert(logits.IsContiguous() &&
           "The Ascend `TopKTopPSamplingFromLogits` provider requires "
           "contiguous logits.");
    assert(out.IsContiguous() &&
           "The Ascend `TopKTopPSamplingFromLogits` provider requires "
           "contiguous output.");
    ValidateHostTensor(top_k);
    ValidateHostTensor(top_p);

    logits_cache_ = ascend::AclTensorCache(logits);
    top_k_cache_ = ascend::AclTensorCache({static_cast<int64_t>(batch_size_)},
                                          ACL_INT32, nullptr);
    top_p_cache_ = ascend::AclTensorCache({static_cast<int64_t>(batch_size_)},
                                          ascend::ToAclDtype(dtype_), nullptr);
    q_cache_ = ascend::AclTensorCache(
        {static_cast<int64_t>(batch_size_), static_cast<int64_t>(vocab_size_)},
        ACL_FLOAT, nullptr);
    selected_idx_cache_ = ascend::AclTensorCache(
        {static_cast<int64_t>(batch_size_)}, ACL_INT64, nullptr);
    selected_logits_cache_ = ascend::AclTensorCache(
        {static_cast<int64_t>(batch_size_), static_cast<int64_t>(vocab_size_)},
        ACL_FLOAT, nullptr);
    out_cache_ = ascend::AclTensorCache(out);
  }

  ~Operator() override {
    if (!ascend::IsAclRuntimeAlive()) return;

    logits_cache_.release();
    top_k_cache_.release();
    top_p_cache_.release();
    q_cache_.release();
    selected_idx_cache_.release();
    selected_logits_cache_.release();
    out_cache_.release();
  }

  void operator()(const Tensor logits, const Tensor top_k, const Tensor top_p,
                  const std::optional<Tensor> indices,
                  const std::string filter_apply_order,
                  const bool deterministic, const bool check_nan,
                  const std::optional<int64_t> seed,
                  const std::optional<int64_t> offset,
                  Tensor out) const override {
    ValidateSupportedOptions(indices, filter_apply_order, deterministic,
                             check_nan);
    assert(logits.IsContiguous() &&
           "The Ascend `TopKTopPSamplingFromLogits` provider requires "
           "contiguous logits.");
    assert(out.IsContiguous() &&
           "The Ascend `TopKTopPSamplingFromLogits` provider requires "
           "contiguous output.");
    ValidateHostTensor(top_k);
    ValidateHostTensor(top_p);

    auto stream = static_cast<aclrtStream>(stream_);
    auto top_k_bytes = batch_size_ * kDataTypeToSize.at(DataType::kInt32);
    auto top_p_bytes = batch_size_ * kDataTypeToSize.at(dtype_);
    auto q_bytes =
        batch_size_ * vocab_size_ * kDataTypeToSize.at(DataType::kFloat32);
    auto selected_idx_bytes =
        batch_size_ * kDataTypeToSize.at(DataType::kInt64);
    auto selected_logits_bytes =
        batch_size_ * vocab_size_ * kDataTypeToSize.at(DataType::kFloat32);

    FillParams(top_k, top_p, seed, offset);

    auto& top_k_arena = ascend::GetWorkspacePool().Ensure(
        stream, top_k_bytes, "top_k_top_p_sampling_from_logits_top_k");
    auto& top_p_arena = ascend::GetWorkspacePool().Ensure(
        stream, top_p_bytes, "top_k_top_p_sampling_from_logits_top_p");
    auto& q_arena = ascend::GetWorkspacePool().Ensure(
        stream, q_bytes, "top_k_top_p_sampling_from_logits_q");
    auto ret = aclrtMemcpy(top_k_arena.buf, top_k_bytes, top_k_host_.data(),
                           top_k_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    assert(ret == ACL_SUCCESS &&
           "Copying `top_k` to the Ascend device failed.");
    ret = aclrtMemcpy(top_p_arena.buf, top_p_bytes, top_p_host_.data(),
                      top_p_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    assert(ret == ACL_SUCCESS &&
           "Copying `top_p` to the Ascend device failed.");
    ret = aclrtMemcpy(q_arena.buf, q_bytes, q_host_.data(), q_bytes,
                      ACL_MEMCPY_HOST_TO_DEVICE);
    assert(ret == ACL_SUCCESS &&
           "Copying sampling noise to the Ascend device failed.");

    auto& selected_idx_arena = ascend::GetWorkspacePool().Ensure(
        stream, selected_idx_bytes, "top_k_top_p_sampling_from_logits_indices");
    auto& selected_logits_arena = ascend::GetWorkspacePool().Ensure(
        stream, selected_logits_bytes,
        "top_k_top_p_sampling_from_logits_selected_logits");

    auto t_logits = logits_cache_.get(const_cast<void*>(logits.data()));
    auto t_top_k = top_k_cache_.get(top_k_arena.buf);
    auto t_top_p = top_p_cache_.get(top_p_arena.buf);
    auto t_q = q_cache_.get(q_arena.buf);
    auto t_selected_idx = selected_idx_cache_.get(selected_idx_arena.buf);
    auto t_selected_logits =
        selected_logits_cache_.get(selected_logits_arena.buf);

    if (!sample_exec_) {
      ret = aclnnTopKTopPSampleGetWorkspaceSize(
          t_logits, t_top_k, t_top_p, t_q, /*eps=*/1e-8,
          /*isNeedLogits=*/false, /*topKGuess=*/32, t_selected_idx,
          t_selected_logits, &sample_ws_size_, &sample_exec_);
      assert(ret == ACL_SUCCESS &&
             "`aclnnTopKTopPSampleGetWorkspaceSize` failed.");
      aclSetAclOpExecutorRepeatable(sample_exec_);
    } else {
      aclSetInputTensorAddr(sample_exec_, 0, t_logits,
                            const_cast<void*>(logits.data()));
      aclSetInputTensorAddr(sample_exec_, 1, t_top_k, top_k_arena.buf);
      aclSetInputTensorAddr(sample_exec_, 2, t_top_p, top_p_arena.buf);
      aclSetInputTensorAddr(sample_exec_, 3, t_q, q_arena.buf);
      aclSetOutputTensorAddr(sample_exec_, 0, t_selected_idx,
                             selected_idx_arena.buf);
      aclSetOutputTensorAddr(sample_exec_, 1, t_selected_logits,
                             selected_logits_arena.buf);
    }

    auto& sample_ws_arena = ascend::GetWorkspacePool().Ensure(
        stream, sample_ws_size_, "top_k_top_p_sampling_from_logits_workspace");
    ret = aclnnTopKTopPSample(sample_ws_arena.buf, sample_ws_size_,
                              sample_exec_, stream);
    assert(ret == ACL_SUCCESS && "`aclnnTopKTopPSample` failed.");

    CastSelectedIdx(selected_idx_arena.buf, out);
  }

 private:
  static void ValidateSupportedOptions(const std::optional<Tensor> indices,
                                       const std::string& filter_apply_order,
                                       const bool deterministic,
                                       const bool check_nan) {
    assert(!indices.has_value() &&
           "The Ascend `TopKTopPSamplingFromLogits` provider does not support "
           "`indices`.");
    assert(filter_apply_order == "top_k_first" &&
           "The Ascend `TopKTopPSamplingFromLogits` provider supports only "
           "`top_k_first`.");
    assert(deterministic &&
           "The Ascend `TopKTopPSamplingFromLogits` provider supports only "
           "the deterministic path.");
    assert(!check_nan &&
           "The Ascend `TopKTopPSamplingFromLogits` provider does not support "
           "`check_nan`.");
  }

  static void ValidateHostTensor(const Tensor tensor) {
    assert(tensor.device().type() == Device::Type::kCpu &&
           "The Ascend `TopKTopPSamplingFromLogits` provider currently "
           "requires host-side `top_k` and `top_p` tensors.");
    assert(tensor.IsContiguous() &&
           "The Ascend `TopKTopPSamplingFromLogits` provider requires "
           "contiguous `top_k` and `top_p` tensors.");
  }

  void CastSelectedIdx(void* selected_idx, Tensor out) const {
    auto stream = static_cast<aclrtStream>(stream_);
    auto t_selected_idx = selected_idx_cache_.get(selected_idx);
    auto t_out = out_cache_.get(out.data());

    if (!cast_exec_) {
      auto ret = aclnnCastGetWorkspaceSize(t_selected_idx, ACL_INT32, t_out,
                                           &cast_ws_size_, &cast_exec_);
      assert(ret == ACL_SUCCESS && "`aclnnCastGetWorkspaceSize` failed.");
      aclSetAclOpExecutorRepeatable(cast_exec_);
    } else {
      aclSetInputTensorAddr(cast_exec_, 0, t_selected_idx, selected_idx);
      aclSetOutputTensorAddr(cast_exec_, 0, t_out, out.data());
    }

    auto& cast_ws_arena = ascend::GetWorkspacePool().Ensure(
        stream, cast_ws_size_,
        "top_k_top_p_sampling_from_logits_cast_workspace");
    auto ret = aclnnCast(cast_ws_arena.buf, cast_ws_size_, cast_exec_, stream);
    assert(ret == ACL_SUCCESS && "`aclnnCast` failed.");
  }

  void FillParams(const Tensor top_k, const Tensor top_p,
                  const std::optional<int64_t> seed,
                  const std::optional<int64_t> offset) const {
    top_k_host_.resize(batch_size_);
    top_p_host_.resize(batch_size_ * kDataTypeToSize.at(dtype_));
    q_host_.resize(batch_size_ * vocab_size_);
    std::exponential_distribution<float> dist(1.0F);
    std::mt19937_64 rng(seed.has_value()
                            ? static_cast<uint64_t>(*seed)
                            : static_cast<uint64_t>(std::random_device{}()));
    rng.discard(static_cast<uint64_t>(offset.value_or(0)));

    for (Tensor::Size row = 0; row < batch_size_; ++row) {
      top_k_host_[row] = static_cast<int32_t>(GetK(top_k, row));
      auto value = static_cast<float>(GetP(top_p, row));
      auto* dst = top_p_host_.data() + row * kDataTypeToSize.at(dtype_);

      if (dtype_ == DataType::kFloat16) {
        auto converted = Float16::FromFloat(value);
        std::memcpy(dst, &converted, sizeof(converted));
      } else {
        auto converted = BFloat16::FromFloat(value);
        std::memcpy(dst, &converted, sizeof(converted));
      }
    }

    for (auto& value : q_host_) value = dist(rng);
  }

  int64_t GetK(const Tensor top_k, Tensor::Size row) const {
    const auto offset = row * top_k.stride(0);
    int64_t value = 0;
    if (top_k.dtype() == DataType::kInt32) {
      value = static_cast<const int32_t*>(top_k.data())[offset];
    } else {
      value = static_cast<const int64_t*>(top_k.data())[offset];
    }

    if (value <= 0) return static_cast<int64_t>(vocab_size_);
    return std::min<int64_t>(value, static_cast<int64_t>(vocab_size_));
  }

  double GetP(const Tensor top_p, Tensor::Size row) const {
    const auto offset = row * top_p.stride(0);
    double value = 1.0;
    switch (top_p.dtype()) {
      case DataType::kFloat16:
        value = static_cast<const Float16*>(top_p.data())[offset].ToFloat();
        break;
      case DataType::kBFloat16:
        value = static_cast<const BFloat16*>(top_p.data())[offset].ToFloat();
        break;
      case DataType::kFloat32:
        value = static_cast<const float*>(top_p.data())[offset];
        break;
      case DataType::kFloat64:
        value = static_cast<const double*>(top_p.data())[offset];
        break;
      default:
        assert(false &&
               "`TopKTopPSamplingFromLogits` received unsupported `top_p` "
               "dtype.");
    }

    if (value <= 0.0 || value > 1.0) return 1.0;
    return value;
  }

  mutable ascend::AclTensorCache logits_cache_;

  mutable ascend::AclTensorCache top_k_cache_;

  mutable ascend::AclTensorCache top_p_cache_;

  mutable ascend::AclTensorCache q_cache_;

  mutable ascend::AclTensorCache selected_idx_cache_;

  mutable ascend::AclTensorCache selected_logits_cache_;

  mutable ascend::AclTensorCache out_cache_;

  mutable std::vector<int32_t> top_k_host_;

  mutable std::vector<std::uint8_t> top_p_host_;

  mutable std::vector<float> q_host_;

  mutable aclOpExecutor* sample_exec_ = nullptr;

  mutable uint64_t sample_ws_size_ = 0;

  mutable aclOpExecutor* cast_exec_ = nullptr;

  mutable uint64_t cast_ws_size_ = 0;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_ASCEND_TOP_K_TOP_P_SAMPLING_FROM_LOGITS_KERNEL_H_
