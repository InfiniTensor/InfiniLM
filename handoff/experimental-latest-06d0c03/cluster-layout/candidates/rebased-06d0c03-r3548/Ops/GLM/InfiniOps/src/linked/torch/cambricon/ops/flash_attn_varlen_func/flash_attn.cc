#include "linked/torch/cambricon/ops/flash_attn_varlen_func/flash_attn.h"

#include <ATen/core/Generator.h>

#include <cassert>
#include <cmath>

#include "common/op_utils/paged_kv_cache.h"
#include "torch/cambricon/c10.h"
#include "torch/tensor_.h"

std::vector<at::Tensor> mha_varlen_fwd(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    std::optional<at::Tensor>& out, const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k, std::optional<at::Tensor>& seqused_k,
    std::optional<at::Tensor>& alibi_slopes, int max_seqlen_q, int max_seqlen_k,
    float dropout_p, float softmax_scale, bool zero_tensors, bool causal,
    int window_size_left, int window_size_right, bool return_softmax,
    std::optional<at::Generator> generator);

namespace infini::ops {

void Operator<FlashAttnVarlenFunc, Device::Type::kCambricon, 16>::operator()(
    const Tensor q, const Tensor k, const Tensor v, const Tensor cu_seqlens_q,
    const Tensor cu_seqlens_k, const std::optional<Tensor> alibi_slopes,
    const std::optional<Tensor> block_table, const int64_t max_seqlen_q,
    const int64_t max_seqlen_k, const double dropout_p,
    const std::optional<double> softmax_scale, const bool causal,
    const std::vector<int64_t> window_size, const double softcap,
    const bool deterministic, const bool return_attn_probs, Tensor out,
    std::optional<Tensor> softmax_lse, std::optional<Tensor> s_dmask) const {
  const C10<Device::Type::kCambricon>::StreamGuard stream_guard{
      C10<Device::Type::kCambricon>::GetStreamFromExternal(stream_,
                                                           device_index_)};
  auto at_q = ToAtenTensor<Device::Type::kCambricon>(
      const_cast<void*>(q.data()), q_shape_, q_strides_, q_dtype_,
      device_index_);
  auto at_k = ToAtenTensor<Device::Type::kCambricon>(
      const_cast<void*>(k.data()), k_shape_, k_strides_, k_dtype_,
      device_index_);
  auto at_v = ToAtenTensor<Device::Type::kCambricon>(
      const_cast<void*>(v.data()), v_shape_, v_strides_, v_dtype_,
      device_index_);
  auto at_cu_seqlens_q = ToAtenTensor<Device::Type::kCambricon>(
      const_cast<void*>(cu_seqlens_q.data()), cu_seqlens_q_shape_,
      cu_seqlens_q_strides_, cu_seqlens_q_dtype_, device_index_);
  auto at_cu_seqlens_k = ToAtenTensor<Device::Type::kCambricon>(
      const_cast<void*>(cu_seqlens_k.data()), cu_seqlens_k_shape_,
      cu_seqlens_k_strides_, cu_seqlens_k_dtype_, device_index_);
  auto at_out = ToAtenTensor<Device::Type::kCambricon>(
      out.data(), out_shape_, out_strides_, out_dtype_, device_index_);

  std::optional<at::Tensor> at_out_optional{at_out};
  std::optional<at::Tensor> at_seqused_k;
  std::optional<at::Tensor> at_alibi_slopes;
  std::optional<at::Generator> generator;
  if (alibi_slopes.has_value()) {
    at_alibi_slopes.emplace(ToAtenTensor<Device::Type::kCambricon>(
        const_cast<void*>(alibi_slopes->data()), alibi_slopes_shape_,
        alibi_slopes_strides_, alibi_slopes_dtype_, device_index_));
  }

  at::Tensor at_k_for_call = at_k;
  at::Tensor at_v_for_call = at_v;
  if (block_table.has_value()) {
    auto at_block_table = ToAtenTensor<Device::Type::kCambricon>(
        const_cast<void*>(block_table->data()), block_table_shape_,
        block_table_strides_, block_table_dtype_, device_index_);
    const auto host_cu_seqlens_k =
        paged_kv_cache::ToHostInt32Vector(at_cu_seqlens_k);
    const auto host_block_table =
        paged_kv_cache::ToHostInt32Vector(at_block_table);
    const int64_t batch_size =
        static_cast<int64_t>(host_cu_seqlens_k.size()) - 1;
    const int64_t table_width = at_block_table.size(1);
    assert(at_block_table.size(0) == batch_size &&
           "KV cache block table batch size does not match cu_seqlens_k");

    std::vector<at::Tensor> packed_k;
    std::vector<at::Tensor> packed_v;
    packed_k.reserve(batch_size);
    packed_v.reserve(batch_size);
    for (int64_t batch = 0; batch < batch_size; ++batch) {
      const int64_t length = static_cast<int64_t>(host_cu_seqlens_k[batch + 1] -
                                                  host_cu_seqlens_k[batch]);
      assert(length >= 0 && "cu_seqlens_k must be nondecreasing");
      packed_k.push_back(paged_kv_cache::GatherSequence(
          at_k, host_block_table, table_width, batch, length));
      packed_v.push_back(paged_kv_cache::GatherSequence(
          at_v, host_block_table, table_width, batch, length));
    }
    at_k_for_call = at::cat(packed_k, 0).contiguous();
    at_v_for_call = at::cat(packed_v, 0).contiguous();
  }

  const auto result = ::mha_varlen_fwd(
      at_q, at_k_for_call, at_v_for_call, at_out_optional, at_cu_seqlens_q,
      at_cu_seqlens_k, at_seqused_k, at_alibi_slopes,
      static_cast<int>(max_seqlen_q), static_cast<int>(max_seqlen_k),
      static_cast<float>(dropout_p),
      static_cast<float>(softmax_scale.value_or(
          1.0 / std::sqrt(static_cast<double>(q_shape_[2])))),
      false, causal, static_cast<int>(window_size[0]),
      static_cast<int>(window_size[1]), false, generator);
  assert(result.size() > 5 &&
         "Cambricon FlashAttention returned an incomplete result");
  at_out.copy_(result[0]);

  if (return_attn_probs) {
    auto at_softmax_lse = ToAtenTensor<Device::Type::kCambricon>(
        softmax_lse->data(), softmax_lse_shape_, softmax_lse_strides_,
        softmax_lse_dtype_, device_index_);
    at_softmax_lse.copy_(result[5]);
  }

  (void)softcap;
  (void)deterministic;
  (void)s_dmask;
}

}  // namespace infini::ops
