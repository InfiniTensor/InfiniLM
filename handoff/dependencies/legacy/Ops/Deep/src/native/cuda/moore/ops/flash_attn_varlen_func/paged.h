#ifndef INFINI_OPS_MOORE_FLASH_ATTN_VARLEN_FUNC_PAGED_H_
#define INFINI_OPS_MOORE_FLASH_ATTN_VARLEN_FUNC_PAGED_H_

#include <cstddef>
#include <cstdint>

#include "data_type.h"

namespace infini::ops {

struct MoorePagedFlashAttnVarlenParams {
  void* out;
  const void* q;
  const void* k_cache;
  const void* v_cache;
  const int32_t* block_table;
  const int32_t* cu_seqlens_q;
  const int32_t* cu_seqlens_k;
  const float* alibi_slopes;

  DataType dtype;
  std::size_t total_q_tokens;
  std::size_t num_heads;
  std::size_t num_kv_heads;
  std::size_t num_seqs;
  std::size_t head_size;
  std::size_t page_size;
  std::size_t max_num_blocks_per_seq;
  std::size_t max_seqlen_q;

  std::ptrdiff_t block_table_batch_stride;
  std::ptrdiff_t q_stride;
  std::ptrdiff_t q_head_stride;
  std::ptrdiff_t k_cache_block_stride;
  std::ptrdiff_t k_cache_slot_stride;
  std::ptrdiff_t k_cache_head_stride;
  std::ptrdiff_t v_cache_block_stride;
  std::ptrdiff_t v_cache_slot_stride;
  std::ptrdiff_t v_cache_head_stride;
  std::ptrdiff_t out_stride;
  std::ptrdiff_t out_head_stride;

  float scale;
  void* stream;
};

void LaunchMoorePagedFlashAttnVarlenFunc(
    const MoorePagedFlashAttnVarlenParams& params);

}  // namespace infini::ops

#endif  // INFINI_OPS_MOORE_FLASH_ATTN_VARLEN_FUNC_PAGED_H_
