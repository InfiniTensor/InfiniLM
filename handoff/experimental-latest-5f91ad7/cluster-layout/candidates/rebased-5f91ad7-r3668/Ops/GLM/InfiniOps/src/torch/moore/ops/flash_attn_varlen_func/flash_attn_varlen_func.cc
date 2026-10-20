#include <ATen/ops/equal.h>

#include <cassert>
#include <cmath>

#include "native/cuda/moore/ops/flash_attn_varlen_func/paged.h"
#include "torch/moore/c10.h"
#include "torch/ops/flash_attn_varlen_func/aten_impl.h"

namespace infini::ops {

namespace detail {

template <>
struct AtenFlashAttnVarlenPolicy<Device::Type::kMoore> {
  static void ValidateWindow(const std::vector<int64_t>& window_size) {
    assert(window_size[0] == -1 && window_size[1] == -1 &&
           "TorchMusa FlashAttention does not support local windows");
  }

  static AtenFlashAttnVarlenOptions MakeOptions(
      bool causal, const std::vector<int64_t>&, const at::Tensor& cu_seqlens_q,
      const at::Tensor& cu_seqlens_k) {
    assert((!causal || at::equal(cu_seqlens_q, cu_seqlens_k)) &&
           "TorchMusa causal FlashAttention requires matching query and "
           "key sequence lengths");

    return {};
  }
};

}  // namespace detail

template class AtenFlashAttnVarlenFunc<Device::Type::kMoore>;

void Operator<FlashAttnVarlenFunc, Device::Type::kMoore, 8>::operator()(
    const Tensor q, const Tensor k, const Tensor v, const Tensor cu_seqlens_q,
    const Tensor cu_seqlens_k, const std::optional<Tensor> alibi_slopes,
    const std::optional<Tensor> block_table, const int64_t max_seqlen_q,
    const int64_t max_seqlen_k, const double dropout_p,
    const std::optional<double> softmax_scale, const bool causal,
    const std::vector<int64_t> window_size, const double softcap,
    const bool deterministic, const bool return_attn_probs, Tensor out,
    std::optional<Tensor> softmax_lse, std::optional<Tensor> s_dmask) const {
  if (!block_table.has_value()) {
    return Base::operator()(q, k, v, cu_seqlens_q, cu_seqlens_k, alibi_slopes,
                            block_table, max_seqlen_q, max_seqlen_k, dropout_p,
                            softmax_scale, causal, window_size, softcap,
                            deterministic, return_attn_probs, out, softmax_lse,
                            s_dmask);
  }

  assert(causal && "Moore paged FlashAttnVarlenFunc requires causal attention");
  assert(window_size[0] == -1 && window_size[1] == -1 &&
         "Moore paged FlashAttnVarlenFunc does not support local windows");
  assert(!return_attn_probs && !softmax_lse.has_value() &&
         !s_dmask.has_value() &&
         "Moore paged FlashAttnVarlenFunc does not return attention "
         "probabilities");
  assert((q_shape_[2] == 64 || q_shape_[2] == 128) &&
         "Moore paged FlashAttnVarlenFunc supports head sizes 64 and 128");
  assert(max_seqlen_k <=
             static_cast<int64_t>(block_table_shape_[1] * k_shape_[1]) &&
         "Moore paged FlashAttnVarlenFunc maximum K length exceeds the "
         "block-table capacity");
  assert((!alibi_slopes.has_value() || alibi_slopes_shape_.size() == 1) &&
         "Moore paged FlashAttnVarlenFunc supports only one-dimensional "
         "ALiBi slopes");

  (void)dropout_p;
  (void)softcap;
  (void)deterministic;

  MoorePagedFlashAttnVarlenParams params{};
  params.out = out.data();
  params.q = q.data();
  params.k_cache = k.data();
  params.v_cache = v.data();
  params.block_table = static_cast<const int32_t*>(block_table->data());
  params.cu_seqlens_q = static_cast<const int32_t*>(cu_seqlens_q.data());
  params.cu_seqlens_k = static_cast<const int32_t*>(cu_seqlens_k.data());
  params.alibi_slopes = alibi_slopes.has_value()
                            ? static_cast<const float*>(alibi_slopes->data())
                            : nullptr;
  params.dtype = q_dtype_;
  params.total_q_tokens = q_shape_[0];
  params.num_heads = q_shape_[1];
  params.num_kv_heads = k_shape_[2];
  params.num_seqs = block_table_shape_[0];
  params.head_size = q_shape_[2];
  params.page_size = k_shape_[1];
  params.max_num_blocks_per_seq = block_table_shape_[1];
  params.max_seqlen_q = max_seqlen_q;
  params.block_table_batch_stride = block_table_strides_[0];
  params.q_stride = q_strides_[0];
  params.q_head_stride = q_strides_[1];
  params.k_cache_block_stride = k_strides_[0];
  params.k_cache_slot_stride = k_strides_[1];
  params.k_cache_head_stride = k_strides_[2];
  params.v_cache_block_stride = v_strides_[0];
  params.v_cache_slot_stride = v_strides_[1];
  params.v_cache_head_stride = v_strides_[2];
  params.out_stride = out_strides_[0];
  params.out_head_stride = out_strides_[1];
  params.scale = static_cast<float>(softmax_scale.value_or(
      1.0 / std::sqrt(static_cast<double>(q_shape_[2]))));
  params.stream = stream_;

  using Backend = C10<Device::Type::kMoore>;
  const typename Backend::StreamGuard stream_guard{
      Backend::GetStreamFromExternal(stream_, device_index_)};

  LaunchMoorePagedFlashAttnVarlenFunc(params);
}

}  // namespace infini::ops
