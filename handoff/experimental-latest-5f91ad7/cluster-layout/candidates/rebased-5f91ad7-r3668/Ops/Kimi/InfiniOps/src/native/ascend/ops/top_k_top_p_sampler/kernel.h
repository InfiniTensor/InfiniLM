#ifndef INFINI_OPS_ASCEND_TOP_K_TOP_P_SAMPLER_KERNEL_H_
#define INFINI_OPS_ASCEND_TOP_K_TOP_P_SAMPLER_KERNEL_H_

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <optional>
#include <random>
#include <vector>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_cast.h"
#include "aclnnop/aclnn_top_k_top_p_sample.h"
#include "base/top_k_top_p_sampler.h"
#include "data_type.h"
#include "native/ascend/common.h"
#include "native/ascend/workspace_pool_.h"
#include "operator.h"
#include "tensor.h"

namespace infini::ops {

template <>
class Operator<TopKTopPSampler, Device::Type::kAscend, 0>
    : public TopKTopPSampler {
 public:
  Operator(const Tensor logits, std::optional<Tensor> k,
           std::optional<Tensor> p, Tensor out)
      : TopKTopPSampler(logits, k, p, out) {
    assert((dtype_ == DataType::kFloat16 || dtype_ == DataType::kBFloat16) &&
           "`TopKTopPSampler` Ascend ACLNN path requires float16 or bfloat16 "
           "logits");
    assert(logits.IsContiguous() &&
           "`TopKTopPSampler` Ascend ACLNN path requires contiguous logits");
    assert(out.IsContiguous() &&
           "`TopKTopPSampler` Ascend ACLNN path requires contiguous output");
    ValidateHostTensor(k);
    ValidateHostTensor(p);

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

  ~Operator() {
    if (!ascend::IsAclRuntimeAlive()) return;

    logits_cache_.release();
    top_k_cache_.release();
    top_p_cache_.release();
    q_cache_.release();
    selected_idx_cache_.release();
    selected_logits_cache_.release();
    out_cache_.release();
  }

  void operator()(const Tensor logits, std::optional<Tensor> k,
                  std::optional<Tensor> p, Tensor out) const override {
    assert(logits.IsContiguous() &&
           "`TopKTopPSampler` Ascend ACLNN path requires contiguous logits");
    assert(out.IsContiguous() &&
           "`TopKTopPSampler` Ascend ACLNN path requires contiguous output");

    auto stream = static_cast<aclrtStream>(stream_);
    auto top_k_bytes = batch_size_ * kDataTypeToSize.at(DataType::kInt32);
    auto top_p_bytes = batch_size_ * kDataTypeToSize.at(dtype_);
    auto q_bytes =
        batch_size_ * vocab_size_ * kDataTypeToSize.at(DataType::kFloat32);
    auto selected_idx_bytes =
        batch_size_ * kDataTypeToSize.at(DataType::kInt64);
    auto selected_logits_bytes =
        batch_size_ * vocab_size_ * kDataTypeToSize.at(DataType::kFloat32);

    FillParams(k, p);

    auto& top_k_arena = ascend::GetWorkspacePool().Ensure(
        stream, top_k_bytes, "top_k_top_p_sample_top_k");
    auto& top_p_arena = ascend::GetWorkspacePool().Ensure(
        stream, top_p_bytes, "top_k_top_p_sample_top_p");
    auto& q_arena = ascend::GetWorkspacePool().Ensure(stream, q_bytes,
                                                      "top_k_top_p_sample_q");
    auto ret = aclrtMemcpy(top_k_arena.buf, top_k_bytes, top_k_host_.data(),
                           top_k_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    assert(ret == ACL_SUCCESS &&
           "`TopKTopPSampler`: copying `top_k` to Ascend failed");
    ret = aclrtMemcpy(top_p_arena.buf, top_p_bytes, top_p_host_.data(),
                      top_p_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    assert(ret == ACL_SUCCESS &&
           "`TopKTopPSampler`: copying `top_p` to Ascend failed");
    ret = aclrtMemcpy(q_arena.buf, q_bytes, q_host_.data(), q_bytes,
                      ACL_MEMCPY_HOST_TO_DEVICE);
    assert(ret == ACL_SUCCESS &&
           "`TopKTopPSampler`: copying `q` to Ascend failed");

    auto& selected_idx_arena = ascend::GetWorkspacePool().Ensure(
        stream, selected_idx_bytes, "top_k_top_p_sample_idx");
    auto& selected_logits_arena = ascend::GetWorkspacePool().Ensure(
        stream, selected_logits_bytes, "top_k_top_p_sample_logits");

    auto t_logits = logits_cache_.get(const_cast<void*>(logits.data()));
    auto t_top_k = top_k_cache_.get(top_k_arena.buf);
    auto t_top_p = top_p_cache_.get(top_p_arena.buf);
    auto t_q = q_cache_.get(q_arena.buf);
    auto t_selected_idx = selected_idx_cache_.get(selected_idx_arena.buf);
    auto t_selected_logits =
        selected_logits_cache_.get(selected_logits_arena.buf);

    if (!sample_exec_) {
      ret = aclnnTopKTopPSampleGetWorkspaceSize(
          t_logits, t_top_k, t_top_p, t_q, /*eps=*/1e-8, /*isNeedLogits=*/false,
          /*topKGuess=*/32, t_selected_idx, t_selected_logits, &sample_ws_size_,
          &sample_exec_);
      assert(ret == ACL_SUCCESS &&
             "`aclnnTopKTopPSampleGetWorkspaceSize` failed");
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
        stream, sample_ws_size_, "top_k_top_p_sample_workspace");
    ret = aclnnTopKTopPSample(sample_ws_arena.buf, sample_ws_size_,
                              sample_exec_, stream);
    assert(ret == ACL_SUCCESS && "`aclnnTopKTopPSample` failed");

    CastSelectedIdx(selected_idx_arena.buf, out);
  }

 private:
  void ValidateHostTensor(std::optional<Tensor> tensor) const {
    if (!tensor.has_value()) return;

    assert(tensor->device().type() == Device::Type::kCpu &&
           "`TopKTopPSampler` Ascend path currently requires host-side "
           "`k`/`p` tensors");
    assert(tensor->IsContiguous() &&
           "`TopKTopPSampler` Ascend path requires contiguous `k`/`p` "
           "tensors");
  }

  void CastSelectedIdx(void* selected_idx, Tensor out) const {
    auto stream = static_cast<aclrtStream>(stream_);
    auto t_selected_idx = selected_idx_cache_.get(selected_idx);
    auto t_out = out_cache_.get(out.data());

    if (!cast_exec_) {
      auto ret = aclnnCastGetWorkspaceSize(t_selected_idx, ACL_INT32, t_out,
                                           &cast_ws_size_, &cast_exec_);
      assert(ret == ACL_SUCCESS && "`aclnnCastGetWorkspaceSize` failed");
      aclSetAclOpExecutorRepeatable(cast_exec_);
    } else {
      aclSetInputTensorAddr(cast_exec_, 0, t_selected_idx, selected_idx);
      aclSetOutputTensorAddr(cast_exec_, 0, t_out, out.data());
    }

    auto& cast_ws_arena = ascend::GetWorkspacePool().Ensure(
        stream, cast_ws_size_, "top_k_top_p_sample_cast_workspace");
    auto ret = aclnnCast(cast_ws_arena.buf, cast_ws_size_, cast_exec_, stream);
    assert(ret == ACL_SUCCESS && "`aclnnCast` failed");
  }

  void FillParams(std::optional<Tensor> k, std::optional<Tensor> p) const {
    top_k_host_.resize(batch_size_);
    top_p_host_.resize(batch_size_ * kDataTypeToSize.at(dtype_));
    q_host_.resize(batch_size_ * vocab_size_);
    std::exponential_distribution<float> dist(1.0F);

    for (Tensor::Size row = 0; row < batch_size_; ++row) {
      top_k_host_[row] = static_cast<int32_t>(GetK(k, row));
      auto value = static_cast<float>(GetP(p, row));
      auto* dst = top_p_host_.data() + row * kDataTypeToSize.at(dtype_);

      if (dtype_ == DataType::kFloat16) {
        auto converted = Float16::FromFloat(value);
        std::memcpy(dst, &converted, sizeof(converted));
      } else {
        auto converted = BFloat16::FromFloat(value);
        std::memcpy(dst, &converted, sizeof(converted));
      }
    }

    // CANN samples by taking argmax of softmax probabilities divided by
    // exponential noise.
    for (auto& value : q_host_) {
      value = dist(rng_);
    }
  }

  int64_t GetK(std::optional<Tensor> k, Tensor::Size row) const {
    if (!k.has_value()) return static_cast<int64_t>(vocab_size_);

    const auto offset = k->size(0) == 1 ? 0 : row;
    int64_t value = 0;
    if (k->dtype() == DataType::kInt32) {
      value = static_cast<const int32_t*>(k->data())[offset];
    } else {
      value = static_cast<const int64_t*>(k->data())[offset];
    }

    if (value <= 0) return static_cast<int64_t>(vocab_size_);
    return std::min<int64_t>(value, static_cast<int64_t>(vocab_size_));
  }

  double GetP(std::optional<Tensor> p, Tensor::Size row) const {
    if (!p.has_value()) return 1.0;

    const auto offset = p->size(0) == 1 ? 0 : row;
    double value = 1.0;
    switch (p->dtype()) {
      case DataType::kFloat16:
        value = static_cast<const Float16*>(p->data())[offset].ToFloat();
        break;
      case DataType::kBFloat16:
        value = static_cast<const BFloat16*>(p->data())[offset].ToFloat();
        break;
      case DataType::kFloat32:
        value = static_cast<const float*>(p->data())[offset];
        break;
      case DataType::kFloat64:
        value = static_cast<const double*>(p->data())[offset];
        break;
      default:
        assert(false && "`TopKTopPSampler` has unsupported `p` dtype");
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

  mutable std::mt19937 rng_{std::random_device{}()};

  mutable aclOpExecutor* sample_exec_ = nullptr;

  mutable uint64_t sample_ws_size_ = 0;

  mutable aclOpExecutor* cast_exec_ = nullptr;

  mutable uint64_t cast_ws_size_ = 0;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_ASCEND_TOP_K_TOP_P_SAMPLER_KERNEL_H_
