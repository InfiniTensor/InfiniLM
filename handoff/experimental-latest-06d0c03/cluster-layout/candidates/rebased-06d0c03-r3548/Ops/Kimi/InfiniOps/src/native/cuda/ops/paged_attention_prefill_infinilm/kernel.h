#ifndef INFINI_OPS_CUDA_PAGED_ATTENTION_PREFILL_INFINILM_KERNEL_H_
#define INFINI_OPS_CUDA_PAGED_ATTENTION_PREFILL_INFINILM_KERNEL_H_

#include <algorithm>
#include <cassert>
#include <cstdint>

#include "base/paged_attention_prefill_infinilm.h"
#include "data_type.h"
#include "dispatcher.h"
#include "native/cuda/kernel_commons.cuh"
#include "native/cuda/ops/paged_attention_prefill_infinilm/kernel.cuh"
#include "native/cuda/runtime_utils.h"

namespace infini::ops {

using PagedAttentionPrefillInfinilmIndexTypes =
    List<DataType::kInt32, DataType::kInt64, DataType::kUInt32>;

template <typename Backend, bool kCumulativeSequenceLengths = false>
class CudaPagedAttentionPrefillInfinilm : public PagedAttentionPrefillInfinilm {
 public:
  using PagedAttentionPrefillInfinilm::PagedAttentionPrefillInfinilm;

  void operator()(const Tensor q, const Tensor k_cache, const Tensor v_cache,
                  const Tensor block_tables, const Tensor seq_lens,
                  const Tensor cum_seq_lens_q,
                  std::optional<Tensor> alibi_slopes, float scale,
                  Tensor out) const override {
    auto cuda_stream =
        static_cast<typename Backend::Stream>(stream_ ? stream_ : 0);

    assert(out.dtype() == dtype_ && q.dtype() == dtype_);
    assert(k_cache.dtype() == dtype_ && v_cache.dtype() == dtype_);
    assert(block_tables.dtype() == index_dtype_ &&
           seq_lens.dtype() == index_dtype_ &&
           cum_seq_lens_q.dtype() == index_dtype_);
    assert(scale == scale_);

    assert((head_size_ == 64 || head_size_ == 128) &&
           "PagedAttentionPrefillInfinilm supports head sizes 64 and 128");

    dim3 grid(static_cast<unsigned>(total_q_tokens_),
              static_cast<unsigned>(num_heads_));

    DispatchFunc<ReducedFloatTypes, PagedAttentionPrefillInfinilmIndexTypes,
                 List<64, 128>>(
        {static_cast<int64_t>(dtype_), static_cast<int64_t>(index_dtype_),
         static_cast<int64_t>(head_size_)},
        [&](auto list_tag) {
          using TData = TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;
          using TIndex =
              TypeMapType<Backend::kDeviceType, ListGet<1>(list_tag)>;
          constexpr int kHeadSize = ListGet<2>(list_tag);
          const auto* sequence_lengths =
              reinterpret_cast<const TIndex*>(seq_lens.data());
          const auto total_kv_lengths = [&]() {
            if constexpr (kCumulativeSequenceLengths) {
              return op::paged_attention_prefill::cuda::
                  CumulativeSequenceLengths<TIndex>{sequence_lengths};
            } else {
              return sequence_lengths;
            }
          }();

          // The HD128 pipeline produces incorrect results on Iluvatar.
          // Keep its native prefill on the global warp kernel.
          if constexpr (kHeadSize == 128 &&
                        Backend::kDeviceType != Device::Type::kIluvatar) {
            if (block_size_ == 256) {
              constexpr int kWarps = 8;
              dim3 pipe_grid(static_cast<unsigned>(num_heads_),
                             static_cast<unsigned>(num_seqs_),
                             static_cast<unsigned>(
                                 (total_q_tokens_ + kWarps - 1) / kWarps));
              PagedAttentionPrefillInfinilmHd128WarpCta8PipeKernel<
                  Backend::kDeviceType, TIndex, TData>
                  <<<pipe_grid, kWarps * 32, 0, cuda_stream>>>(
                      reinterpret_cast<TData*>(out.data()),
                      reinterpret_cast<const TData*>(q.data()),
                      reinterpret_cast<const TData*>(k_cache.data()),
                      reinterpret_cast<const TData*>(v_cache.data()),
                      reinterpret_cast<const TIndex*>(block_tables.data()),
                      total_kv_lengths,
                      reinterpret_cast<const TIndex*>(cum_seq_lens_q.data()),
                      alibi_slopes.has_value()
                          ? reinterpret_cast<const float*>(alibi_slopes->data())
                          : nullptr,
                      num_kv_heads_, scale, max_num_blocks_per_seq_,
                      block_size_, block_table_batch_stride_, q_stride_,
                      q_head_stride_, k_cache_block_stride_,
                      k_cache_slot_stride_, k_cache_head_stride_,
                      v_cache_block_stride_, v_cache_slot_stride_,
                      v_cache_head_stride_, out_stride_, out_head_stride_);
            } else {
              dim3 legacy_grid(static_cast<unsigned>(num_heads_),
                               static_cast<unsigned>(total_q_tokens_));
              op::paged_attention_prefill::cuda::
                  PagedAttentionPrefillWarpGlobalKernel<
                      Backend::kDeviceType, TIndex, TData, kHeadSize>
                  <<<legacy_grid, 32, 0, cuda_stream>>>(
                      reinterpret_cast<TData*>(out.data()),
                      reinterpret_cast<const TData*>(q.data()),
                      reinterpret_cast<const TData*>(k_cache.data()),
                      reinterpret_cast<const TData*>(v_cache.data()),
                      reinterpret_cast<const TIndex*>(block_tables.data()),
                      total_kv_lengths,
                      reinterpret_cast<const TIndex*>(cum_seq_lens_q.data()),
                      alibi_slopes.has_value()
                          ? reinterpret_cast<const float*>(alibi_slopes->data())
                          : nullptr,
                      num_heads_, num_seqs_, num_kv_heads_, total_q_tokens_,
                      scale, max_num_blocks_per_seq_, block_size_,
                      block_table_batch_stride_, q_stride_, q_head_stride_,
                      k_cache_block_stride_, k_cache_slot_stride_,
                      k_cache_head_stride_, v_cache_block_stride_,
                      v_cache_slot_stride_, v_cache_head_stride_, out_stride_,
                      out_head_stride_);
            }
          } else {
            dim3 legacy_grid(static_cast<unsigned>(num_heads_),
                             static_cast<unsigned>(total_q_tokens_));
            if constexpr (Backend::kDeviceType == Device::Type::kIluvatar) {
              // The kernel folds grid.z into its token index when grid.y
              // would exceed CUDA's 65535 limit (e.g. B16 x I4096).
              if (total_q_tokens_ > 65535) {
                legacy_grid.y = 65535;
                legacy_grid.z =
                    static_cast<unsigned>((total_q_tokens_ + 65534) / 65535);
              }
            }
            op::paged_attention_prefill::cuda::
                PagedAttentionPrefillWarpGlobalKernel<Backend::kDeviceType,
                                                      TIndex, TData, kHeadSize>
                <<<legacy_grid, 32, 0, cuda_stream>>>(
                    reinterpret_cast<TData*>(out.data()),
                    reinterpret_cast<const TData*>(q.data()),
                    reinterpret_cast<const TData*>(k_cache.data()),
                    reinterpret_cast<const TData*>(v_cache.data()),
                    reinterpret_cast<const TIndex*>(block_tables.data()),
                    total_kv_lengths,
                    reinterpret_cast<const TIndex*>(cum_seq_lens_q.data()),
                    alibi_slopes.has_value()
                        ? reinterpret_cast<const float*>(alibi_slopes->data())
                        : nullptr,
                    num_heads_, num_seqs_, num_kv_heads_, total_q_tokens_,
                    scale, max_num_blocks_per_seq_, block_size_,
                    block_table_batch_stride_, q_stride_, q_head_stride_,
                    k_cache_block_stride_, k_cache_slot_stride_,
                    k_cache_head_stride_, v_cache_block_stride_,
                    v_cache_slot_stride_, v_cache_head_stride_, out_stride_,
                    out_head_stride_);
          }
        },
        "CudaPagedAttentionPrefillInfinilm::operator()");
  }
};

}  // namespace infini::ops

#endif
