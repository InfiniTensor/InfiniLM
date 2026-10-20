#ifndef INFINI_OPS_CUDA_PAGED_ATTENTION_INFINILM_KERNEL_H_
#define INFINI_OPS_CUDA_PAGED_ATTENTION_INFINILM_KERNEL_H_

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>

#include "base/paged_attention_infinilm.h"
#include "data_type.h"
#include "dispatcher.h"
#include "native/cuda/kernel_commons.cuh"
#include "native/cuda/ops/paged_attention_infinilm/kernel.cuh"
#include "native/cuda/runtime_utils.h"

namespace infini::ops {

using PagedAttentionInfinilmIndexTypes =
    List<DataType::kInt32, DataType::kInt64, DataType::kUInt32>;

static constexpr int kMaxSplits = 8;
static constexpr int kDefaultNumSplits = 4;
static constexpr int kLongSequenceNumSplits = 8;
static constexpr std::size_t kLongSequenceMinBlocks = 32;
static constexpr std::size_t kLowBatchMaxNumSeqs = 4;

template <typename Backend>
class CudaPagedAttentionInfinilm : public PagedAttentionInfinilm {
 public:
  using PagedAttentionInfinilm::PagedAttentionInfinilm;

  std::size_t workspace_size_in_bytes() const override {
    return static_cast<std::size_t>(kMaxSplits) * num_seqs_ * num_heads_ *
           (head_size_ + 2) * sizeof(float);
  }

  void operator()(const Tensor q, const Tensor k_cache, const Tensor v_cache,
                  const Tensor block_tables, const Tensor seq_lens,
                  std::optional<Tensor> alibi_slopes, float scale,
                  Tensor out) const override {
    auto cuda_stream =
        static_cast<typename Backend::Stream>(stream_ ? stream_ : 0);

    assert(out.dtype() == dtype_ && q.dtype() == dtype_);
    assert(k_cache.dtype() == dtype_ && v_cache.dtype() == dtype_);
    assert(block_tables.dtype() == index_dtype_ &&
           seq_lens.dtype() == index_dtype_);
    assert(scale == scale_);

    assert((head_size_ == 64 || head_size_ == 128) &&
           "PagedAttentionInfinilm supports head sizes 64 and 128");

    dim3 grid(static_cast<unsigned>(num_heads_),
              static_cast<unsigned>(num_seqs_));

    DispatchFunc<ReducedFloatTypes, PagedAttentionInfinilmIndexTypes,
                 List<64, 128>>(
        {static_cast<int64_t>(dtype_), static_cast<int64_t>(index_dtype_),
         static_cast<int64_t>(head_size_)},
        [&](auto list_tag) {
          using TData = TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;
          using TIndex =
              TypeMapType<Backend::kDeviceType, ListGet<1>(list_tag)>;
          constexpr int kHeadSize = ListGet<2>(list_tag);

          // Low-batch long-context decode benefits from more KV parallelism.
          // High-batch runs already expose enough work and pay extra combine
          // overhead, so keep the default split count there.
          const int num_splits =
              max_num_blocks_per_seq_ >= kLongSequenceMinBlocks &&
                      num_seqs_ <= kLowBatchMaxNumSeqs
                  ? kLongSequenceNumSplits
                  : kDefaultNumSplits;
          const std::size_t n = num_seqs_ * num_heads_;
          const std::size_t acc_elems =
              static_cast<std::size_t>(kMaxSplits) * n * kHeadSize;
          const std::size_t m_elems = static_cast<std::size_t>(kMaxSplits) * n;
          const std::size_t l_elems = static_cast<std::size_t>(kMaxSplits) * n;
          const std::size_t needed_bytes =
              (acc_elems + m_elems + l_elems) * sizeof(float);

          if constexpr (kHeadSize == 128) {
            if (workspace_ != nullptr &&
                workspace_size_in_bytes_ >= needed_bytes) {
              float* workspace = static_cast<float*>(workspace_);
              float* partial_acc = workspace;
              float* partial_m = partial_acc + acc_elems;
              float* partial_l = partial_m + m_elems;
              dim3 grid_split(static_cast<unsigned>(num_heads_),
                              static_cast<unsigned>(num_seqs_),
                              static_cast<unsigned>(num_splits));

              PagedAttentionInfinilmSplitKvCtaKernel<TIndex, TData, kHeadSize>
                  <<<grid_split, 64, 0, cuda_stream>>>(
                      partial_acc, partial_m, partial_l,
                      reinterpret_cast<const TData*>(q.data()),
                      reinterpret_cast<const TData*>(k_cache.data()),
                      reinterpret_cast<const TData*>(v_cache.data()),
                      reinterpret_cast<const TIndex*>(block_tables.data()),
                      reinterpret_cast<const TIndex*>(seq_lens.data()),
                      alibi_slopes.has_value()
                          ? reinterpret_cast<const float*>(alibi_slopes->data())
                          : nullptr,
                      num_kv_heads_, scale, max_num_blocks_per_seq_,
                      block_size_, q_stride_, k_cache_block_stride_,
                      k_cache_slot_stride_, k_cache_head_stride_,
                      v_cache_block_stride_, v_cache_slot_stride_,
                      v_cache_head_stride_, num_splits);
              PagedAttentionInfinilmSplitKvCombineKernel<Backend::kDeviceType,
                                                         TData, kHeadSize>
                  <<<grid, 32, 0, cuda_stream>>>(
                      reinterpret_cast<TData*>(out.data()), partial_acc,
                      partial_m, partial_l, num_splits, out_stride_);
            } else {
              PagedAttentionInfinilmDecodeWarpKernel<TIndex, TData, kHeadSize>
                  <<<grid, 32, 0, cuda_stream>>>(
                      reinterpret_cast<TData*>(out.data()),
                      reinterpret_cast<const TData*>(q.data()),
                      reinterpret_cast<const TData*>(k_cache.data()),
                      reinterpret_cast<const TData*>(v_cache.data()),
                      reinterpret_cast<const TIndex*>(block_tables.data()),
                      reinterpret_cast<const TIndex*>(seq_lens.data()),
                      alibi_slopes.has_value()
                          ? reinterpret_cast<const float*>(alibi_slopes->data())
                          : nullptr,
                      num_heads_, num_kv_heads_, scale, max_num_blocks_per_seq_,
                      block_size_, k_cache_block_stride_, k_cache_head_stride_,
                      k_cache_slot_stride_, v_cache_block_stride_,
                      v_cache_head_stride_, v_cache_slot_stride_, q_stride_,
                      q_head_stride_, out_stride_, out_head_stride_,
                      block_table_batch_stride_, seq_lens_stride_);
            }
          } else {
            PagedAttentionInfinilmDecodeWarpKernel<TIndex, TData, kHeadSize>
                <<<grid, 32, 0, cuda_stream>>>(
                    reinterpret_cast<TData*>(out.data()),
                    reinterpret_cast<const TData*>(q.data()),
                    reinterpret_cast<const TData*>(k_cache.data()),
                    reinterpret_cast<const TData*>(v_cache.data()),
                    reinterpret_cast<const TIndex*>(block_tables.data()),
                    reinterpret_cast<const TIndex*>(seq_lens.data()),
                    alibi_slopes.has_value()
                        ? reinterpret_cast<const float*>(alibi_slopes->data())
                        : nullptr,
                    num_heads_, num_kv_heads_, scale, max_num_blocks_per_seq_,
                    block_size_, k_cache_block_stride_, k_cache_head_stride_,
                    k_cache_slot_stride_, v_cache_block_stride_,
                    v_cache_head_stride_, v_cache_slot_stride_, q_stride_,
                    q_head_stride_, out_stride_, out_head_stride_,
                    block_table_batch_stride_, seq_lens_stride_);
          }
        },
        "CudaPagedAttentionInfinilm::operator()");
  }
};

}  // namespace infini::ops

#endif
