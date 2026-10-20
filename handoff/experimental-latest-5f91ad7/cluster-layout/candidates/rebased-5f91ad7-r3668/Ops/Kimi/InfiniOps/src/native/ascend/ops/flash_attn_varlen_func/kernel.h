#ifndef INFINI_OPS_ASCEND_FLASH_ATTN_VARLEN_FUNC_KERNEL_H_
#define INFINI_OPS_ASCEND_FLASH_ATTN_VARLEN_FUNC_KERNEL_H_

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_fused_infer_attention_score_v4.h"
#include "base/flash_attn_varlen_func.h"
#include "native/ascend/common.h"
#include "native/ascend/workspace_pool_.h"
#include "operator.h"

namespace infini::ops {

template <>
class Operator<FlashAttnVarlenFunc, Device::Type::kAscend>
    : public FlashAttnVarlenFunc {
 public:
  Operator(const Tensor q, const Tensor k, const Tensor v,
           const Tensor cu_seqlens_q, const Tensor cu_seqlens_k,
           const int64_t max_seqlen_q, const int64_t max_seqlen_k, Tensor out)
      : Operator(q, k, v, cu_seqlens_q, cu_seqlens_k, std::nullopt,
                 std::nullopt, max_seqlen_q, max_seqlen_k, 0.0, std::nullopt,
                 false, {-1, -1}, 0.0, false, false, out, std::nullopt,
                 std::nullopt) {}

  Operator(const Tensor q, const Tensor k, const Tensor v,
           const Tensor cu_seqlens_q, const Tensor cu_seqlens_k,
           const std::optional<Tensor> alibi_slopes,
           const std::optional<Tensor> block_table, const int64_t max_seqlen_q,
           const int64_t max_seqlen_k, const double dropout_p,
           const std::optional<double> softmax_scale, const bool causal,
           const std::vector<int64_t> window_size, const double softcap,
           const bool deterministic, const bool return_attn_probs, Tensor out,
           std::optional<Tensor> softmax_lse, std::optional<Tensor> s_dmask)
      : FlashAttnVarlenFunc(q, k, v, cu_seqlens_q, cu_seqlens_k, alibi_slopes,
                            block_table, max_seqlen_q, max_seqlen_k, dropout_p,
                            softmax_scale, causal, window_size, softcap,
                            deterministic, return_attn_probs, out, softmax_lse,
                            s_dmask),
        sequence_count_(cu_seqlens_q.numel() - 1),
        q_cache_(q),
        out_cache_(out) {
    ValidateSupportedOptions(alibi_slopes, return_attn_probs, softmax_lse,
                             s_dmask);
    ValidateTensors(q, k, v, block_table, out);

    cu_i32_host_.resize(sequence_count_ + 1);
    q_lengths_i64_.resize(sequence_count_);
    k_lengths_i64_.resize(sequence_count_);

    if (causal || window_size != std::vector<int64_t>({-1, -1})) {
      InitializeAttentionMask();
    }
  }

  ~Operator() override {
    if (!ascend::IsAclRuntimeAlive()) return;

    attention_mask_cache_.destroy();
    if (attention_mask_data_) aclrtFree(attention_mask_data_);
  }

  using FlashAttnVarlenFunc::operator();

  // CANN paged attention consumes a BnBsH cache. InfiniOps exposes the same
  // storage as BnBsND, so flatten the head dimensions in the ACL descriptor.
  static aclTensor* BuildPagedCacheAclTensor(const Tensor& tensor) {
    const std::vector<int64_t> shape{
        static_cast<int64_t>(tensor.size(0)),
        static_cast<int64_t>(tensor.size(1)),
        static_cast<int64_t>(tensor.size(2) * tensor.size(3)),
    };
    std::vector<int64_t> strides(shape.size());
    int64_t stride = 1;
    for (int64_t index = static_cast<int64_t>(shape.size()) - 1; index >= 0;
         --index) {
      strides[index] = stride;
      stride *= shape[index];
    }
    const std::vector<int64_t> storage_shape{stride};
    return aclCreateTensor(shape.data(), static_cast<int64_t>(shape.size()),
                           ascend::ToAclDtype(tensor.dtype()), strides.data(),
                           /*storageOffset=*/0, ACL_FORMAT_ND,
                           storage_shape.data(),
                           static_cast<int64_t>(storage_shape.size()),
                           const_cast<void*>(tensor.data()));
  }

  void operator()(const Tensor q, const Tensor k, const Tensor v,
                  const Tensor cu_seqlens_q, const Tensor cu_seqlens_k,
                  const std::optional<Tensor> alibi_slopes,
                  const std::optional<Tensor> block_table,
                  const int64_t max_seqlen_q, const int64_t max_seqlen_k,
                  const double dropout_p,
                  const std::optional<double> softmax_scale, const bool causal,
                  const std::vector<int64_t> window_size, const double softcap,
                  const bool deterministic, const bool return_attn_probs,
                  Tensor out, std::optional<Tensor> softmax_lse,
                  std::optional<Tensor> s_dmask) const override {
    ValidateSupportedOptions(alibi_slopes, return_attn_probs, softmax_lse,
                             s_dmask);
    ValidateTensors(q, k, v, block_table, out);
    const bool paged = block_table.has_value();

    auto stream = static_cast<aclrtStream>(stream_);
    auto actual_seq_lengths =
        MakeSequenceLengths(cu_seqlens_q, q_lengths_i64_, stream,
                            /*cumulative=*/true);
    auto actual_seq_lengths_kv =
        MakeSequenceLengths(cu_seqlens_k, k_lengths_i64_, stream,
                            /*cumulative=*/!block_table.has_value());
    auto t_q = q_cache_.get(const_cast<void*>(q.data()));
    auto t_k = paged ? BuildPagedCacheAclTensor(k) : ascend::BuildAclTensor(k);
    auto t_v = paged ? BuildPagedCacheAclTensor(v) : ascend::BuildAclTensor(v);
    auto t_out = out_cache_.get(out.data());
    auto t_attention_mask =
        attention_mask_data_ ? attention_mask_cache_.get(attention_mask_data_)
                             : nullptr;
    aclTensor* t_block_table = nullptr;
    if (block_table.has_value()) {
      t_block_table = ascend::BuildAclTensor(*block_table);
    }
    const aclTensor* keys[] = {t_k};
    const aclTensor* values[] = {t_v};
    auto key_list = aclCreateTensorList(keys, 1);
    auto value_list = aclCreateTensorList(values, 1);

    const auto max_token_count =
        static_cast<int64_t>(std::numeric_limits<int32_t>::max());
    auto pre_tokens = max_token_count;
    auto next_tokens = max_token_count;
    auto sparse_mode = int64_t{0};

    if (causal || window_size != std::vector<int64_t>({-1, -1})) {
      sparse_mode = window_size == std::vector<int64_t>({-1, -1}) ? 3 : 4;
      if (window_size[0] >= 0) pre_tokens = window_size[0];
      next_tokens = causal ? 0 : window_size[1];
      if (next_tokens < 0) next_tokens = max_token_count;
    }

    auto scale = softmax_scale.value_or(1.0 / std::sqrt(q.size(2)));
    const auto num_key_value_heads = paged ? k.size(2) : k.size(1);
    const auto block_size = paged ? k.size(1) : 0;
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnFusedInferAttentionScoreV4GetWorkspaceSize(
        t_q, key_list, value_list,
        /*pseShift=*/nullptr, t_attention_mask, actual_seq_lengths,
        actual_seq_lengths_kv,
        /*deqScale1=*/nullptr,
        /*quantScale1=*/nullptr,
        /*deqScale2=*/nullptr,
        /*quantScale2=*/nullptr,
        /*quantOffset2=*/nullptr,
        /*antiquantScale=*/nullptr,
        /*antiquantOffset=*/nullptr,
        /*blockTable=*/t_block_table,
        /*queryPaddingSize=*/nullptr,
        /*kvPaddingSize=*/nullptr,
        /*keyAntiquantScale=*/nullptr,
        /*keyAntiquantOffset=*/nullptr,
        /*valueAntiquantScale=*/nullptr,
        /*valueAntiquantOffset=*/nullptr,
        /*keySharedPrefix=*/nullptr,
        /*valueSharedPrefix=*/nullptr,
        /*actualSharedPrefixLen=*/nullptr,
        /*queryRope=*/nullptr,
        /*keyRope=*/nullptr,
        /*keyRopeAntiquantScale=*/nullptr,
        /*dequantScaleQuery=*/nullptr,
        /*learnableSink=*/nullptr, q.size(1), scale, pre_tokens, next_tokens,
        input_layout_.data(), num_key_value_heads, sparse_mode,
        // Correct fully masked rows when right-down causal has Q longer than
        // KV.
        /*innerPrecise=*/2,
        /*blockSize=*/block_size,
        /*antiquantMode=*/0,
        /*softmaxLseFlag=*/false,
        /*keyAntiquantMode=*/0,
        /*valueAntiquantMode=*/0,
        /*queryQuantMode=*/0, t_out,
        /*softmaxLse=*/nullptr, &workspace_size, &executor);
    assert(ret == ACL_SUCCESS &&
           "Ascend `FlashAttnVarlenFunc` workspace query failed");

    auto& arena = ascend::GetWorkspacePool().Ensure(
        stream, workspace_size, "flash_attn_varlen_func_workspace");
    ret = aclnnFusedInferAttentionScoreV4(arena.buf, workspace_size, executor,
                                          stream);
    assert(ret == ACL_SUCCESS &&
           "Ascend `FlashAttnVarlenFunc` execution failed");

    aclDestroyTensorList(key_list);
    aclDestroyTensorList(value_list);
    if (t_block_table) aclDestroyTensor(t_block_table);
    aclDestroyIntArray(actual_seq_lengths);
    aclDestroyIntArray(actual_seq_lengths_kv);

    (void)max_seqlen_q;
    (void)max_seqlen_k;
    (void)dropout_p;
    (void)softcap;
    (void)deterministic;
  }

 private:
  static void ValidateSupportedOptions(
      const std::optional<Tensor>& alibi_slopes, bool return_attn_probs,
      const std::optional<Tensor>& softmax_lse,
      const std::optional<Tensor>& s_dmask) {
    assert(!alibi_slopes.has_value() &&
           "Ascend FlashAttnVarlenFunc does not support ALiBi");
    assert(!return_attn_probs && !softmax_lse.has_value() &&
           !s_dmask.has_value() &&
           "Ascend `FlashAttnVarlenFunc` does not support auxiliary outputs");
  }

  static void ValidateTensors(const Tensor q, const Tensor k, const Tensor v,
                              const std::optional<Tensor>& block_table,
                              const Tensor out) {
    const bool paged = block_table.has_value();
    assert(q.IsContiguous() && k.IsContiguous() && v.IsContiguous() &&
           out.IsContiguous() &&
           "Ascend FlashAttnVarlenFunc requires contiguous tensors");
    assert(q.ndim() == 3 && out.ndim() == 3 &&
           ((paged && k.ndim() == 4 && v.ndim() == 4) ||
            (!paged && k.ndim() == 3 && v.ndim() == 3)) &&
           "Ascend FlashAttnVarlenFunc expects dense TND key/value or "
           "paged block key/value tensors");
    if (paged) {
      assert(block_table->ndim() == 2 && block_table->IsContiguous() &&
             block_table->dtype() == DataType::kInt32 && k.size(1) > 0 &&
             "Ascend paged attention requires a contiguous int32 block "
             "table and a positive cache block size");
    }
  }

  void InitializeAttentionMask() {
    constexpr int64_t mask_size = 2048;
    std::vector<uint8_t> mask(mask_size * mask_size);

    for (int64_t row = 0; row < mask_size; ++row) {
      for (int64_t column = row + 1; column < mask_size; ++column) {
        mask[row * mask_size + column] = 1;
      }
    }

    const auto bytes = mask.size() * sizeof(mask[0]);
    auto ret =
        aclrtMalloc(&attention_mask_data_, bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
    assert(ret == ACL_SUCCESS &&
           "Ascend `FlashAttnVarlenFunc` failed to allocate attention mask");
    ret = aclrtMemcpy(attention_mask_data_, bytes, mask.data(), bytes,
                      ACL_MEMCPY_HOST_TO_DEVICE);
    assert(ret == ACL_SUCCESS &&
           "Ascend `FlashAttnVarlenFunc` failed to upload attention mask");
    attention_mask_cache_ = ascend::AclTensorCache(
        {mask_size, mask_size}, ACL_BOOL, attention_mask_data_);
  }

  // Dense TND K/V uses cumulative endpoints, while paged K/V uses per-batch
  // lengths. Query is always dense TND and therefore cumulative.
  aclIntArray* MakeSequenceLengths(const Tensor cu_seqlens,
                                   std::vector<int64_t>& lengths,
                                   aclrtStream stream, bool cumulative) const {
    const auto bytes = cu_i32_host_.size() * sizeof(cu_i32_host_[0]);
    auto ret = aclrtMemcpyAsync(cu_i32_host_.data(), bytes, cu_seqlens.data(),
                                bytes, ACL_MEMCPY_DEVICE_TO_HOST, stream);
    assert(ret == ACL_SUCCESS &&
           "Ascend `FlashAttnVarlenFunc` failed to copy sequence lengths");
    ret = aclrtSynchronizeStream(stream);
    assert(ret == ACL_SUCCESS &&
           "Ascend `FlashAttnVarlenFunc` failed to synchronize lengths");

    std::transform(
        cu_i32_host_.begin() + 1, cu_i32_host_.end(), cu_i32_host_.begin(),
        lengths.begin(), [cumulative](int32_t current, int32_t previous) {
          return static_cast<int64_t>(cumulative ? current
                                                 : current - previous);
        });
    return aclCreateIntArray(lengths.data(), lengths.size());
  }

  size_t sequence_count_{0};
  mutable ascend::AclTensorCache q_cache_;
  mutable ascend::AclTensorCache out_cache_;
  mutable ascend::AclTensorCache attention_mask_cache_;
  mutable std::vector<int32_t> cu_i32_host_;
  mutable std::vector<int64_t> q_lengths_i64_;
  mutable std::vector<int64_t> k_lengths_i64_;
  void* attention_mask_data_{nullptr};
  mutable std::array<char, 4> input_layout_{'T', 'N', 'D', '\0'};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_ASCEND_FLASH_ATTN_VARLEN_FUNC_KERNEL_H_
