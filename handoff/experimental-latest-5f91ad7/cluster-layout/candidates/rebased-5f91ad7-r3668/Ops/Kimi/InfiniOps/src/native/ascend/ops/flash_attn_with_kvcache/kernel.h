#ifndef INFINI_OPS_ASCEND_FLASH_ATTN_WITH_KVCACHE_KERNEL_H_
#define INFINI_OPS_ASCEND_FLASH_ATTN_WITH_KVCACHE_KERNEL_H_

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_incre_flash_attention_v4.h"
#include "base/flash_attn_with_kvcache.h"
#include "native/ascend/common.h"
#include "native/ascend/workspace_pool_.h"
#include "operator.h"

namespace infini::ops {

template <>
class Operator<FlashAttnWithKvcache, Device::Type::kAscend>
    : public FlashAttnWithKvcache {
 public:
  Operator(const Tensor q, Tensor k_cache, Tensor v_cache, Tensor out)
      : Operator(q, k_cache, v_cache, std::nullopt, std::nullopt, std::nullopt,
                 std::nullopt, std::optional<Tensor>{}, std::nullopt,
                 std::nullopt, std::nullopt, std::nullopt, std::nullopt, false,
                 {-1, -1}, 0.0, true, 0, false, out, std::nullopt) {}

  Operator(const Tensor q, Tensor k_cache, Tensor v_cache,
           const std::optional<Tensor> k, const std::optional<Tensor> v,
           const std::optional<Tensor> rotary_cos,
           const std::optional<Tensor> rotary_sin, const int64_t cache_seqlens,
           const std::optional<Tensor> cache_batch_idx,
           const std::optional<Tensor> cache_leftpad,
           const std::optional<Tensor> block_table,
           const std::optional<Tensor> alibi_slopes,
           const std::optional<double> softmax_scale, const bool causal,
           const std::vector<int64_t> window_size, const double softcap,
           const bool rotary_interleaved, const int64_t num_splits,
           const bool return_softmax_lse, Tensor out,
           std::optional<Tensor> softmax_lse)
      : FlashAttnWithKvcache(q, k_cache, v_cache, k, v, rotary_cos, rotary_sin,
                             cache_seqlens, cache_batch_idx, cache_leftpad,
                             block_table, alibi_slopes, softmax_scale, causal,
                             window_size, softcap, rotary_interleaved,
                             num_splits, return_softmax_lse, out, softmax_lse) {
    Initialize(q, k_cache, v_cache, k, v, rotary_cos, rotary_sin,
               cache_batch_idx, cache_leftpad, block_table, alibi_slopes,
               window_size, softcap, num_splits, return_softmax_lse, out,
               softmax_lse);
  }

  Operator(const Tensor q, Tensor k_cache, Tensor v_cache,
           const std::optional<Tensor> k, const std::optional<Tensor> v,
           const std::optional<Tensor> rotary_cos,
           const std::optional<Tensor> rotary_sin,
           const std::optional<Tensor> cache_seqlens,
           const std::optional<Tensor> cache_batch_idx,
           const std::optional<Tensor> cache_leftpad,
           const std::optional<Tensor> block_table,
           const std::optional<Tensor> alibi_slopes,
           const std::optional<double> softmax_scale, const bool causal,
           const std::vector<int64_t> window_size, const double softcap,
           const bool rotary_interleaved, const int64_t num_splits,
           const bool return_softmax_lse, Tensor out,
           std::optional<Tensor> softmax_lse)
      : FlashAttnWithKvcache(q, k_cache, v_cache, k, v, rotary_cos, rotary_sin,
                             cache_seqlens, cache_batch_idx, cache_leftpad,
                             block_table, alibi_slopes, softmax_scale, causal,
                             window_size, softcap, rotary_interleaved,
                             num_splits, return_softmax_lse, out, softmax_lse) {
    Initialize(q, k_cache, v_cache, k, v, rotary_cos, rotary_sin,
               cache_batch_idx, cache_leftpad, block_table, alibi_slopes,
               window_size, softcap, num_splits, return_softmax_lse, out,
               softmax_lse);
  }

  using FlashAttnWithKvcache::operator();

  void operator()(const Tensor q, Tensor k_cache, Tensor v_cache,
                  const std::optional<Tensor> k, const std::optional<Tensor> v,
                  const std::optional<Tensor> rotary_cos,
                  const std::optional<Tensor> rotary_sin,
                  const int64_t cache_seqlens,
                  const std::optional<Tensor> cache_batch_idx,
                  const std::optional<Tensor> cache_leftpad,
                  const std::optional<Tensor> block_table,
                  const std::optional<Tensor> alibi_slopes,
                  const std::optional<double> softmax_scale, const bool causal,
                  const std::vector<int64_t> window_size, const double softcap,
                  const bool rotary_interleaved, const int64_t num_splits,
                  const bool return_softmax_lse, Tensor out,
                  std::optional<Tensor> softmax_lse) const override {
    ValidateRuntimeArguments(k, v, rotary_cos, rotary_sin, cache_batch_idx,
                             cache_leftpad, alibi_slopes, window_size, softcap,
                             num_splits, return_softmax_lse, softmax_lse);
    Run(q, k_cache, v_cache, std::nullopt, cache_seqlens, block_table,
        softmax_scale, out);
    (void)causal;
    (void)rotary_interleaved;
  }

  void operator()(const Tensor q, Tensor k_cache, Tensor v_cache,
                  const std::optional<Tensor> k, const std::optional<Tensor> v,
                  const std::optional<Tensor> rotary_cos,
                  const std::optional<Tensor> rotary_sin,
                  const std::optional<Tensor> cache_seqlens,
                  const std::optional<Tensor> cache_batch_idx,
                  const std::optional<Tensor> cache_leftpad,
                  const std::optional<Tensor> block_table,
                  const std::optional<Tensor> alibi_slopes,
                  const std::optional<double> softmax_scale, const bool causal,
                  const std::vector<int64_t> window_size, const double softcap,
                  const bool rotary_interleaved, const int64_t num_splits,
                  const bool return_softmax_lse, Tensor out,
                  std::optional<Tensor> softmax_lse) const override {
    ValidateRuntimeArguments(k, v, rotary_cos, rotary_sin, cache_batch_idx,
                             cache_leftpad, alibi_slopes, window_size, softcap,
                             num_splits, return_softmax_lse, softmax_lse);
    Run(q, k_cache, v_cache, cache_seqlens, std::nullopt, block_table,
        softmax_scale, out);
    (void)causal;
    (void)rotary_interleaved;
  }

 private:
  static Tensor BnsdView(const Tensor tensor) {
    auto shape = Tensor::Shape{tensor.shape()};
    auto strides = Tensor::Strides{tensor.strides()};
    std::swap(shape[1], shape[2]);
    std::swap(strides[1], strides[2]);
    return Tensor{const_cast<void*>(tensor.data()), shape, tensor.dtype(),
                  tensor.device(), strides};
  }

  void Initialize(const Tensor q, const Tensor k_cache, const Tensor v_cache,
                  const std::optional<Tensor>& k,
                  const std::optional<Tensor>& v,
                  const std::optional<Tensor>& rotary_cos,
                  const std::optional<Tensor>& rotary_sin,
                  const std::optional<Tensor>& cache_batch_idx,
                  const std::optional<Tensor>& cache_leftpad,
                  const std::optional<Tensor>& block_table,
                  const std::optional<Tensor>& alibi_slopes,
                  const std::vector<int64_t>& window_size, double softcap,
                  int64_t num_splits, bool return_softmax_lse, const Tensor out,
                  const std::optional<Tensor>& softmax_lse) {
    assert(q.size(1) == 1 &&
           "Ascend `FlashAttnWithKvcache` supports single-token decode");
    assert(q.IsContiguous() && k_cache.IsContiguous() &&
           v_cache.IsContiguous() && out.IsContiguous() &&
           "Ascend `FlashAttnWithKvcache` requires contiguous tensors");
    assert(!k.has_value() && !v.has_value() && !rotary_cos.has_value() &&
           !rotary_sin.has_value() && !cache_batch_idx.has_value() &&
           !cache_leftpad.has_value() && !alibi_slopes.has_value() &&
           "Ascend `FlashAttnWithKvcache` does not support optional cache "
           "updates, rotary inputs, cache remapping, left padding, or ALiBi");
    assert(window_size == std::vector<int64_t>({-1, -1}) && softcap == 0.0 &&
           num_splits == 0 && !return_softmax_lse && !softmax_lse.has_value() &&
           "Ascend `FlashAttnWithKvcache` does not support windowing, "
           "softcap, split execution, or softmax LSE output");

    paged_ = block_table.has_value();
    auto q_view = paged_ ? BnsdView(q) : q;
    auto out_view = paged_ ? BnsdView(out) : out;
    query_cache_ = ascend::AclTensorCache(q_view);
    out_cache_ = ascend::AclTensorCache(out_view);
    if (paged_) block_table_cache_ = ascend::AclTensorCache(*block_table);
    seq_i32_host_.resize(batch_size_);
    seq_i64_host_.resize(batch_size_);
  }

  void Run(const Tensor q, const Tensor k_cache, const Tensor v_cache,
           const std::optional<Tensor> cache_seqlens,
           const std::optional<int64_t> scalar_cache_seqlens,
           const std::optional<Tensor> block_table,
           const std::optional<double> softmax_scale, Tensor out) const {
    auto stream = static_cast<aclrtStream>(stream_);
    assert(block_table.has_value() == paged_ &&
           "Ascend `FlashAttnWithKvcache` cache mode changed between calls");
    auto actual_seq_lengths =
        MakeSeqLengths(cache_seqlens, scalar_cache_seqlens, stream);
    assert(!paged_ || actual_seq_lengths != nullptr &&
                          "Ascend paged attention requires sequence lengths");

    auto max_cache_length = static_cast<int64_t>(
        paged_ ? block_table->size(1) * k_cache.size(1) : k_cache.size(1));
    assert(std::all_of(seq_i64_host_.begin(), seq_i64_host_.end(),
                       [&](int64_t length) {
                         return length >= 0 && length <= max_cache_length;
                       }) &&
           "Ascend `FlashAttnWithKvcache` received invalid sequence lengths");

    auto query_view = paged_ ? BnsdView(q) : q;
    auto key_view = paged_ ? BnsdView(k_cache) : k_cache;
    auto value_view = paged_ ? BnsdView(v_cache) : v_cache;
    auto out_view = paged_ ? BnsdView(out) : out;
    auto t_query = query_cache_.get(const_cast<void*>(query_view.data()));
    auto t_key = ascend::BuildAclTensor(key_view);
    auto t_value = ascend::BuildAclTensor(value_view);
    auto t_out = out_cache_.get(out_view.data());
    aclTensor* t_block_table = nullptr;
    if (block_table.has_value()) {
      t_block_table =
          block_table_cache_.get(const_cast<void*>(block_table->data()));
    }

    const aclTensor* key_tensors[] = {t_key};
    const aclTensor* value_tensors[] = {t_value};
    auto key_list = aclCreateTensorList(key_tensors, 1);
    auto value_list = aclCreateTensorList(value_tensors, 1);
    assert(key_list && value_list &&
           "Ascend `FlashAttnWithKvcache` failed to create tensor lists");

    auto scale = softmax_scale.value_or(
        1.0 / std::sqrt(static_cast<double>(head_size_)));
    auto num_heads = static_cast<int64_t>(q.size(2));
    auto num_kv_heads = static_cast<int64_t>(k_cache.size(2));
    auto block_size = paged_ ? static_cast<int64_t>(k_cache.size(1)) : 0;
    auto* layout = paged_ ? bnsd_layout_.data() : bsnd_layout_.data();

    aclOpExecutor* executor = nullptr;
    uint64_t workspace_size = 0;
    auto ret = aclnnIncreFlashAttentionV4GetWorkspaceSize(
        t_query, key_list, value_list, nullptr, nullptr, actual_seq_lengths,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        t_block_table, nullptr, num_heads, scale, layout, num_kv_heads,
        block_size, 0, t_out, &workspace_size, &executor);
    assert(ret == ACL_SUCCESS &&
           "Ascend `FlashAttnWithKvcache` workspace query failed");

    auto& arena = ascend::GetWorkspacePool().Ensure(stream, workspace_size);
    ret =
        aclnnIncreFlashAttentionV4(arena.buf, workspace_size, executor, stream);
    assert(ret == ACL_SUCCESS &&
           "Ascend `FlashAttnWithKvcache` execution failed");

    aclDestroyTensorList(key_list);
    aclDestroyTensorList(value_list);
    if (actual_seq_lengths) aclDestroyIntArray(actual_seq_lengths);
  }

  aclIntArray* MakeSeqLengths(
      const std::optional<Tensor>& cache_seqlens,
      const std::optional<int64_t>& scalar_cache_seqlens,
      aclrtStream stream) const {
    if (scalar_cache_seqlens.has_value()) {
      std::fill(seq_i64_host_.begin(), seq_i64_host_.end(),
                *scalar_cache_seqlens);
    } else if (cache_seqlens.has_value()) {
      auto bytes = seq_i32_host_.size() * sizeof(seq_i32_host_[0]);
      auto ret =
          aclrtMemcpyAsync(seq_i32_host_.data(), bytes, cache_seqlens->data(),
                           bytes, ACL_MEMCPY_DEVICE_TO_HOST, stream);
      assert(ret == ACL_SUCCESS &&
             "Ascend `FlashAttnWithKvcache` failed to copy sequence lengths");
      ret = aclrtSynchronizeStream(stream);
      assert(ret == ACL_SUCCESS &&
             "Ascend `FlashAttnWithKvcache` failed to synchronize lengths");
      std::transform(
          seq_i32_host_.begin(), seq_i32_host_.end(), seq_i64_host_.begin(),
          [](int32_t length) { return static_cast<int64_t>(length); });
    } else {
      return nullptr;
    }

    return aclCreateIntArray(seq_i64_host_.data(), seq_i64_host_.size());
  }

  static void ValidateRuntimeArguments(
      const std::optional<Tensor>& k, const std::optional<Tensor>& v,
      const std::optional<Tensor>& rotary_cos,
      const std::optional<Tensor>& rotary_sin,
      const std::optional<Tensor>& cache_batch_idx,
      const std::optional<Tensor>& cache_leftpad,
      const std::optional<Tensor>& alibi_slopes,
      const std::vector<int64_t>& window_size, double softcap,
      int64_t num_splits, bool return_softmax_lse,
      const std::optional<Tensor>& softmax_lse) {
    assert(!k.has_value() && !v.has_value() && !rotary_cos.has_value() &&
           !rotary_sin.has_value() && !cache_batch_idx.has_value() &&
           !cache_leftpad.has_value() && !alibi_slopes.has_value() &&
           "Ascend `FlashAttnWithKvcache` received an unsupported optional "
           "tensor");
    assert(window_size == std::vector<int64_t>({-1, -1}) && softcap == 0.0 &&
           num_splits == 0 && !return_softmax_lse && !softmax_lse.has_value() &&
           "Ascend `FlashAttnWithKvcache` received unsupported options");
  }

  bool paged_{false};
  mutable ascend::AclTensorCache query_cache_;
  mutable ascend::AclTensorCache out_cache_;
  mutable ascend::AclTensorCache block_table_cache_;
  mutable std::vector<int32_t> seq_i32_host_;
  mutable std::vector<int64_t> seq_i64_host_;
  mutable std::array<char, 5> bsnd_layout_{'B', 'S', 'N', 'D', '\0'};
  mutable std::array<char, 5> bnsd_layout_{'B', 'N', 'S', 'D', '\0'};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_ASCEND_FLASH_ATTN_WITH_KVCACHE_KERNEL_H_
