#include "kernel.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "native/cuda/nvidia/ops/flash_common.h"
#include "native/cuda/nvidia/ops/flash_head_dim.h"
#include "native/cuda/nvidia/ops/flash_workspace.h"
#include "native/cuda/nvidia/ops/infini_flash_cuda.h"

namespace infini::ops {
namespace {

using native_flash::DeviceScope;
using native_flash::plain_layout;
using native_flash::vector_aligned;

bool int_aligned(Tensor t) {
  return !t.data() ||
         reinterpret_cast<std::uintptr_t>(t.data()) % alignof(int) == 0;
}

std::size_t checked_product(std::initializer_list<std::size_t> sizes) {
  std::size_t result = 1;
  for (auto size : sizes) {
    if (size && result > std::numeric_limits<std::size_t>::max() / size)
      throw std::overflow_error("native varlen Attention scratch overflow");
    result *= size;
  }
  return result;
}

}  // namespace

struct Operator<FlashAttnVarlenFunc, Device::Type::kNvidia, 0>::Impl {
  infini_flash::Flash_fwd_params params{};
  std::size_t workspace_bytes = 0;
  std::size_t lse_offset = 0;
  std::optional<double> scale;
  bool requested_causal;
  std::array<int64_t, 2> requested_window;
  bool paged;
  int64_t max_q, max_k;
};

#define VARLEN_ARGS                                                          \
  const Tensor q, const Tensor k, const Tensor v, const Tensor cu_seqlens_q, \
      const Tensor cu_seqlens_k, const std::optional<Tensor> alibi_slopes,   \
      const std::optional<Tensor> block_table, const int64_t max_seqlen_q,   \
      const int64_t max_seqlen_k, const double dropout_p,                    \
      const std::optional<double> softmax_scale, const bool causal,          \
      const std::vector<int64_t> window_size, const double softcap,          \
      const bool deterministic, const bool return_attn_probs, Tensor out,    \
      std::optional<Tensor> softmax_lse, std::optional<Tensor> s_dmask
#define VARLEN_VALUES                                               \
  q, k, v, cu_seqlens_q, cu_seqlens_k, alibi_slopes, block_table,   \
      max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, causal, \
      window_size, softcap, deterministic, return_attn_probs, out,  \
      softmax_lse, s_dmask

namespace {
bool supports_metadata(VARLEN_ARGS) {
  if (!FlashAttnVarlenFunc::supports(VARLEN_VALUES)) return false;
  const bool paged = block_table.has_value();
  const bool valid_window =
      ((window_size[0] == -1 && window_size[1] == -1) ||
       (window_size[0] >= 0 &&
        window_size[0] <= std::numeric_limits<int>::max() - 127 &&
        window_size[1] == 0));
  if (!plain_layout(q, 3) || !plain_layout(out, 3) ||
      !plain_layout(k, paged ? 4 : 3) || !plain_layout(v, paged ? 4 : 3) ||
      !native_flash::head_dim_bin(q.size(2)) ||
      q.device().type() != Device::Type::kNvidia || !vector_aligned(q) ||
      !vector_aligned(k) || !vector_aligned(v) || !vector_aligned(out) ||
      !plain_layout(cu_seqlens_q, 1) || !plain_layout(cu_seqlens_k, 1) ||
      !int_aligned(cu_seqlens_q) || !int_aligned(cu_seqlens_k) ||
      max_seqlen_q > std::numeric_limits<int>::max() - 127 ||
      max_seqlen_k > std::numeric_limits<int>::max() - 127 ||
      max_seqlen_q > q.size(0) || alibi_slopes || return_attn_probs ||
      !valid_window)
    return false;
  if (paged) {
    if (!plain_layout(*block_table, 2) || !int_aligned(*block_table) ||
        block_table->size(1) >
            (std::numeric_limits<int>::max() - 127) / k.size(1) ||
        max_seqlen_k > block_table->size(1) * k.size(1))
      return false;
  } else if (max_seqlen_k > k.size(0)) {
    return false;
  }
  (void)softmax_scale;
  (void)causal;
  return true;
}
}  // namespace

bool Operator<FlashAttnVarlenFunc, Device::Type::kNvidia, 0>::supports(
    VARLEN_ARGS) {
  if (!supports_metadata(VARLEN_VALUES)) return false;
  // These native kernels require SM80+. The attribute query allocates no
  // scratch.
  int major = 0;
  return cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                                q.device().index()) == cudaSuccess &&
         major >= 8;
}

void Operator<FlashAttnVarlenFunc, Device::Type::kNvidia, 0>::prepare(
    VARLEN_ARGS) {
  (void)alibi_slopes;
  (void)dropout_p;
  (void)softcap;
  (void)deterministic;
  (void)return_attn_probs;
  (void)softmax_lse;
  (void)s_dmask;
  DeviceScope device(device_index_);
  int major;
  INFINI_FLASH_CUDA_CHECK(cudaDeviceGetAttribute(
      &major, cudaDevAttrComputeCapabilityMajor, device_index_));
  if (major < 8)
    throw std::invalid_argument("native varlen Attention requires SM80+");
  auto state = std::make_shared<Impl>();
  state->scale = softmax_scale;
  state->requested_causal = causal;
  state->requested_window = {window_size[0], window_size[1]};
  state->paged = block_table.has_value();
  state->max_q = max_seqlen_q;
  state->max_k = max_seqlen_k;
  auto& p = state->params;
  // Follow pinned flash_api.cpp set_params_fprop and mha_varlen_fwd. The
  // upstream single-token GQA reshape is an optional optimization, omitted.
  p.b = cu_seqlens_q.size(0) - 1;
  p.h = q.size(1);
  p.h_k = k.size(k.ndim() - 2);
  p.h_h_k_ratio = p.h / p.h_k;
  p.d = q.size(2);
  p.d_rounded = native_flash::head_dim_bin(p.d);
  p.seqlen_q = max_seqlen_q;
  p.seqlen_k = max_seqlen_k;
  p.seqlen_q_rounded = ((p.seqlen_q + 127) / 128) * 128;
  p.seqlen_k_rounded = ((p.seqlen_k + 127) / 128) * 128;
  p.total_q = q.size(0);
  p.q_row_stride = q.stride(0);
  p.q_head_stride = q.stride(1);
  p.k_row_stride = k.stride(k.ndim() - 3);
  p.k_head_stride = k.stride(k.ndim() - 2);
  p.v_row_stride = v.stride(v.ndim() - 3);
  p.v_head_stride = v.stride(v.ndim() - 2);
  p.o_row_stride = out.stride(0);
  p.o_head_stride = out.stride(1);
  p.k_batch_stride = state->paged ? k.stride(0) : 0;
  p.v_batch_stride = state->paged ? v.stride(0) : 0;
  p.block_table_batch_stride = state->paged ? block_table->stride(0) : 0;
  p.page_block_size = state->paged ? k.size(1) : 1;
  p.is_bf16 = q.dtype() == DataType::kBFloat16;
  const float scale = static_cast<float>(
      softmax_scale.value_or(1.0 / std::sqrt(static_cast<double>(p.d))));
  if (!std::isfinite(scale))
    throw std::invalid_argument("nonfinite varlen Attention scale");
  p.scale_softmax = scale;
  p.scale_softmax_log2 = scale * M_LOG2E;
  p.p_dropout = p.rp_dropout = 1.f;
  p.p_dropout_in_uint8_t = 255;
  p.scale_softmax_rp_dropout = scale;
  const bool finite_window = window_size[0] >= 0;
  // FA's causal specialization ignores the left bound. Finite causal windows
  // must use its existing local specialization, including single-token Q.
  p.is_causal = !finite_window && causal && max_seqlen_q != 1;
  p.window_size_left = finite_window ? static_cast<int>(window_size[0])
                                     : (p.is_causal ? max_seqlen_k : -1);
  p.window_size_right = finite_window ? 0 : (p.is_causal ? 0 : -1);
  p.is_seqlens_k_cumulative = true;
  p.unpadded_lse = true;
  p.num_splits = 0;  // Paged KV still uses upstream split-KV kernel, one split.
  native_flash::WorkspaceLayout layout;
  state->lse_offset = layout.append(
      checked_product({static_cast<std::size_t>(p.h),
                       static_cast<std::size_t>(p.total_q), sizeof(float)}));
  state->workspace_bytes = layout.required();
  impl_ = std::move(state);
}

std::size_t Operator<FlashAttnVarlenFunc, Device::Type::kNvidia,
                     0>::workspace_size_in_bytes() const {
  return impl_->workspace_bytes;
}

Operator<FlashAttnVarlenFunc, Device::Type::kNvidia, 0>::~Operator() {
  impl_.reset();
}

void Operator<FlashAttnVarlenFunc, Device::Type::kNvidia, 0>::operator()(
    VARLEN_ARGS) const {
  if (!supports_metadata(VARLEN_VALUES) || q.shape() != q_shape_ ||
      q.strides() != q_strides_ || k.shape() != k_shape_ ||
      k.strides() != k_strides_ || v.shape() != v_shape_ ||
      v.strides() != v_strides_ || out.shape() != out_shape_ ||
      out.strides() != out_strides_ ||
      cu_seqlens_q.shape() != cu_seqlens_q_shape_ ||
      cu_seqlens_q.strides() != cu_seqlens_q_strides_ ||
      cu_seqlens_k.shape() != cu_seqlens_k_shape_ ||
      cu_seqlens_k.strides() != cu_seqlens_k_strides_ ||
      (block_table ? Tensor::Shape{block_table->shape()} : Tensor::Shape{}) !=
          block_table_shape_ ||
      (block_table ? Tensor::Strides{block_table->strides()}
                   : Tensor::Strides{}) != block_table_strides_ ||
      q.device().index() != device_index_ || q.dtype() != q_dtype_ ||
      softmax_scale != impl_->scale || causal != impl_->requested_causal ||
      window_size[0] != impl_->requested_window[0] ||
      window_size[1] != impl_->requested_window[1] ||
      max_seqlen_q != impl_->max_q || max_seqlen_k != impl_->max_k)
    throw std::invalid_argument(
        "native varlen Attention prepared metadata changed");
  DeviceScope device(device_index_);
  auto stream = static_cast<cudaStream_t>(stream_);
  native_flash::WorkspaceLease scratch(workspace_, workspace_size_in_bytes_,
                                       impl_->workspace_bytes, stream,
                                       device_index_);
  auto p = impl_->params;
  p.softmax_lse_ptr =
      static_cast<std::byte*>(scratch.data()) + impl_->lse_offset;
  p.q_ptr = const_cast<void*>(q.data());
  p.k_ptr = const_cast<void*>(k.data());
  p.v_ptr = const_cast<void*>(v.data());
  p.o_ptr = out.data();
  // The pinned FA parameter ABI uses mutable pointers, but inference reads
  // cumulative lengths and the page table without modifying them.
  p.cu_seqlens_q =
      const_cast<int*>(static_cast<const int*>(cu_seqlens_q.data()));
  p.cu_seqlens_k =
      const_cast<int*>(static_cast<const int*>(cu_seqlens_k.data()));
  p.block_table =
      block_table
          ? const_cast<int*>(static_cast<const int*>(block_table->data()))
          : nullptr;
  // Pinned flash_api.cpp forces split-KV for paged KV; non-split kernels do
  // not read block_table. This one-split case needs no accumulation scratch.
  if (impl_->paged)
    native_flash::launch_splitkv(p, stream);
  else
    native_flash::launch_varlen(p, stream);
  scratch.finish();
}

#undef VARLEN_ARGS
#undef VARLEN_VALUES
}  // namespace infini::ops
