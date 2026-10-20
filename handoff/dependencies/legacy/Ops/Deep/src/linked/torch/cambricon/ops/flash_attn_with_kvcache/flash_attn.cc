#include "linked/torch/cambricon/ops/flash_attn_with_kvcache/flash_attn.h"

#include <ATen/core/Generator.h>

#include <algorithm>
#include <cassert>
#include <cstdint>

#include "common/op_utils/paged_kv_cache.h"

std::vector<at::Tensor> mha_varlen_fwd(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    std::optional<at::Tensor>& out, const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k, std::optional<at::Tensor>& seqused_k,
    std::optional<at::Tensor>& alibi_slopes, int max_seqlen_q, int max_seqlen_k,
    float dropout_p, float softmax_scale, bool zero_tensors, bool causal,
    int window_size_left, int window_size_right, bool return_softmax,
    std::optional<at::Generator> generator);

namespace infini::ops::linked::torch::cambricon {
namespace {

namespace paged_kv_cache = ::infini::ops::paged_kv_cache;
using paged_kv_cache::ToHostInt32Vector;

at::Tensor ToDeviceIndices(const std::vector<int32_t>& values,
                           const at::Device& device) {
  auto cpu = at::from_blob(const_cast<int32_t*>(values.data()),
                           {static_cast<int64_t>(values.size())},
                           at::TensorOptions().dtype(at::kInt))
                 .clone();
  return cpu.to(device);
}

void UpdatePagedCache(const at::Tensor& cache, const at::Tensor& values,
                      const std::vector<int32_t>& block_table,
                      int64_t table_width, int64_t batch, int64_t offset) {
  const int64_t page_size = cache.size(1);
  int64_t source_offset = 0;
  while (source_offset < values.size(1)) {
    const int64_t logical_offset = offset + source_offset;
    const int64_t table_column = logical_offset / page_size;
    const int64_t page_offset = logical_offset % page_size;
    assert(table_column < table_width && "KV cache block table is too small");
    const int64_t block = block_table[batch * table_width + table_column];
    const int64_t count =
        std::min(page_size - page_offset, values.size(1) - source_offset);
    cache.select(0, block)
        .slice(0, page_offset, page_offset + count)
        .copy_(values.select(0, batch).slice(0, source_offset,
                                             source_offset + count));
    source_offset += count;
  }
}

}  // namespace

std::vector<at::Tensor> FlashAttnKvcache::Call(
    at::Tensor& q, const at::Tensor& k_cache, const at::Tensor& v_cache,
    std::optional<const at::Tensor>& k, std::optional<const at::Tensor>& v,
    std::optional<const at::Tensor>& cache_seqlens,
    std::optional<const at::Tensor>& rotary_cos,
    std::optional<const at::Tensor>& rotary_sin,
    std::optional<const at::Tensor>& cache_batch_idx,
    std::optional<const at::Tensor>& cache_leftpad,
    std::optional<at::Tensor>& block_table,
    std::optional<at::Tensor>& alibi_slopes, std::optional<at::Tensor>& out,
    float softmax_scale, bool causal, int window_size_left,
    int window_size_right, float softcap, bool rotary_interleaved,
    int num_splits) {
  assert(!rotary_cos.has_value() && !rotary_sin.has_value() &&
         "Cambricon KV-cache attention does not yet support rotary tables");
  assert(!cache_leftpad.has_value() &&
         "Cambricon KV-cache attention does not yet support left padding");
  assert(softcap == 0.0f &&
         "Cambricon KV-cache attention does not support softcap");

  const int64_t batch_size = q.size(0);
  const int64_t query_length = q.size(1);
  const int64_t num_heads = q.size(2);
  const int64_t head_size = q.size(3);
  const bool paged = block_table.has_value();
  const int64_t append_length = k.has_value() ? k->size(1) : 0;
  assert(!k.has_value() || cache_seqlens.has_value());
  assert(!paged || !cache_batch_idx.has_value());

  std::vector<int32_t> lengths(
      batch_size, static_cast<int32_t>(paged ? 0 : k_cache.size(1)));
  if (cache_seqlens.has_value()) {
    lengths = ToHostInt32Vector(*cache_seqlens);
  }
  std::vector<int32_t> cache_rows(batch_size);
  for (int64_t i = 0; i < batch_size; ++i) {
    cache_rows[i] = static_cast<int32_t>(i);
  }
  if (cache_batch_idx.has_value()) {
    cache_rows = ToHostInt32Vector(*cache_batch_idx);
  }

  std::vector<int32_t> host_block_table;
  int64_t table_width = 0;
  if (paged) {
    host_block_table = ToHostInt32Vector(*block_table);
    table_width = block_table->size(1);
  }

  if (k.has_value()) {
    for (int64_t batch = 0; batch < batch_size; ++batch) {
      if (paged) {
        UpdatePagedCache(k_cache, *k, host_block_table, table_width, batch,
                         lengths[batch]);
        UpdatePagedCache(v_cache, *v, host_block_table, table_width, batch,
                         lengths[batch]);
      } else {
        const int64_t row = cache_rows[batch];
        k_cache.select(0, row)
            .slice(0, lengths[batch], lengths[batch] + append_length)
            .copy_(k->select(0, batch));
        v_cache.select(0, row)
            .slice(0, lengths[batch], lengths[batch] + append_length)
            .copy_(v->select(0, batch));
      }
    }
  }

  std::vector<at::Tensor> packed_k;
  std::vector<at::Tensor> packed_v;
  std::vector<int32_t> cu_seqlens_q{0};
  std::vector<int32_t> cu_seqlens_k{0};
  int64_t max_key_length = 0;
  packed_k.reserve(batch_size);
  packed_v.reserve(batch_size);
  for (int64_t batch = 0; batch < batch_size; ++batch) {
    const int64_t length = lengths[batch] + append_length;
    assert(length > 0 && "KV-cache attention requires a non-empty cache");
    max_key_length = std::max(max_key_length, length);
    cu_seqlens_q.push_back(cu_seqlens_q.back() + query_length);
    cu_seqlens_k.push_back(cu_seqlens_k.back() + length);
    if (paged) {
      packed_k.push_back(paged_kv_cache::GatherSequence(
          k_cache, host_block_table, table_width, batch, length));
      packed_v.push_back(paged_kv_cache::GatherSequence(
          v_cache, host_block_table, table_width, batch, length));
    } else {
      const int64_t row = cache_rows[batch];
      packed_k.push_back(k_cache.select(0, row).slice(0, 0, length));
      packed_v.push_back(v_cache.select(0, row).slice(0, 0, length));
    }
  }

  auto packed_q =
      q.contiguous().view({batch_size * query_length, num_heads, head_size});
  auto at_cu_seqlens_q = ToDeviceIndices(cu_seqlens_q, q.device());
  auto at_cu_seqlens_k = ToDeviceIndices(cu_seqlens_k, q.device());
  auto at_packed_k = at::cat(packed_k, 0).contiguous();
  auto at_packed_v = at::cat(packed_v, 0).contiguous();
  std::optional<at::Tensor> packed_out;
  std::optional<at::Tensor> seqused_k;
  std::optional<at::Generator> generator;
  const auto result = ::mha_varlen_fwd(
      packed_q, at_packed_k, at_packed_v, packed_out, at_cu_seqlens_q,
      at_cu_seqlens_k, seqused_k, alibi_slopes, static_cast<int>(query_length),
      static_cast<int>(max_key_length), 0.0f, softmax_scale, false, causal,
      window_size_left, window_size_right, false, generator);
  assert(result.size() > 5 &&
         "Cambricon FlashAttention returned an incomplete result");

  auto result_out = result[0].view(q.sizes());
  auto result_lse = result[5]
                        .view({num_heads, batch_size, query_length})
                        .permute({1, 0, 2})
                        .contiguous();
  (void)out;
  (void)rotary_interleaved;
  (void)num_splits;
  return {result_out, result_lse};
}

}  // namespace infini::ops::linked::torch::cambricon

namespace infini::ops::linked::torch {

template class TorchFlashAttnWithKvcache<
    ::infini::ops::linked::torch::cambricon::FlashAttnKvcache>;

}  // namespace infini::ops::linked::torch
