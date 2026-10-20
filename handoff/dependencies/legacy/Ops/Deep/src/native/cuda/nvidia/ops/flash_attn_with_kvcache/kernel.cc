#include "kernel.h"
#include "infini_flash_cuda.h"
#include "flash.h"
#include "num_splits.h"
#include "native/cuda/nvidia/ops/flash_workspace.h"
#include "native/cuda/nvidia/ops/flash_head_dim.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <cstdio>
#include <stdexcept>

namespace infini::ops::native_flash {
void launch_32_fp16_false(infini_flash::Flash_fwd_params&, cudaStream_t);
void launch_32_fp16_true(infini_flash::Flash_fwd_params&, cudaStream_t);
void launch_32_bf16_false(infini_flash::Flash_fwd_params&, cudaStream_t);
void launch_32_bf16_true(infini_flash::Flash_fwd_params&, cudaStream_t);
void launch_64_fp16_false(infini_flash::Flash_fwd_params&, cudaStream_t);
void launch_64_fp16_true(infini_flash::Flash_fwd_params&, cudaStream_t);
void launch_64_bf16_false(infini_flash::Flash_fwd_params&, cudaStream_t);
void launch_64_bf16_true(infini_flash::Flash_fwd_params&, cudaStream_t);
void launch_96_fp16_false(infini_flash::Flash_fwd_params&, cudaStream_t);
void launch_96_fp16_true(infini_flash::Flash_fwd_params&, cudaStream_t);
void launch_96_bf16_false(infini_flash::Flash_fwd_params&, cudaStream_t);
void launch_96_bf16_true(infini_flash::Flash_fwd_params&, cudaStream_t);
void launch_128_fp16_false(infini_flash::Flash_fwd_params&, cudaStream_t);
void launch_128_fp16_true(infini_flash::Flash_fwd_params&, cudaStream_t);
void launch_128_bf16_false(infini_flash::Flash_fwd_params&, cudaStream_t);
void launch_128_bf16_true(infini_flash::Flash_fwd_params&, cudaStream_t);
void launch_192_fp16_false(infini_flash::Flash_fwd_params&, cudaStream_t);
void launch_192_fp16_true(infini_flash::Flash_fwd_params&, cudaStream_t);
void launch_192_bf16_false(infini_flash::Flash_fwd_params&, cudaStream_t);
void launch_192_bf16_true(infini_flash::Flash_fwd_params&, cudaStream_t);
void launch_256_fp16_false(infini_flash::Flash_fwd_params&, cudaStream_t);
void launch_256_fp16_true(infini_flash::Flash_fwd_params&, cudaStream_t);
void launch_256_bf16_false(infini_flash::Flash_fwd_params&, cudaStream_t);
void launch_256_bf16_true(infini_flash::Flash_fwd_params&, cudaStream_t);
}  // namespace infini::ops::native_flash

namespace infini::ops {
namespace {
struct DeviceScope {
  int previous;
  explicit DeviceScope(int device) {
    INFINI_FLASH_CUDA_CHECK(cudaGetDevice(&previous));
    INFINI_FLASH_CUDA_CHECK(cudaSetDevice(device));
  }
  ~DeviceScope() { cudaSetDevice(previous); }
};

// Candidate ordinary MHA-only extension. Keep true d/strides, use existing 96
// specialization's Is_even_K=false masking. No physical padding or new algorithm.
int ordinary_head_dim_bin(int dimension) {
  return dimension == 72 || dimension == 80 ? 96 : native_flash::head_dim_bin(dimension);
}

bool plain_layout(Tensor t, int dimensions) {
  if (dimensions < 1 || dimensions > 4 || t.ndim() != dimensions ||
      t.stride(-1) != 1) return false;
  std::array<int, 4> axes{};
  int count = 0;
  for (int axis = 0; axis < dimensions; ++axis) {
    if (t.size(axis) <= 0 || t.size(axis) > std::numeric_limits<int>::max() - 127 ||
        t.stride(axis) < 0)
      return false;
    // A singleton axis is never indexed away from zero. LM unsqueeze views
    // legitimately use stride zero here; this is not a broadcasted data axis.
    if (t.size(axis) == 1) continue;
    axes[count++] = axis;
  }
  // Logical dimension order need not match storage order. Static KV is a
  // head-major view; the upstream kernel already receives its row/head strides.
  std::sort(axes.begin(), axes.begin() + count,
            [&](int a, int b) { return t.stride(a) < t.stride(b); });
  std::size_t extent = 1;
  for (int index = 0; index < count; ++index) {
    const int axis = axes[index];
    const auto size = static_cast<std::size_t>(t.size(axis));
    const auto stride = static_cast<std::size_t>(t.stride(axis));
    if (stride < extent || (size - 1) >
        (std::numeric_limits<std::size_t>::max() - extent) / stride) return false;
    extent += (size - 1) * stride;
  }
  return true;
}

std::size_t product(std::initializer_list<std::size_t> sizes) {
  std::size_t result = 1;
  for (auto size : sizes) {
    if (size && result > std::numeric_limits<std::size_t>::max() / size)
      throw std::overflow_error("native Attention scratch size overflow");
    result *= size;
  }
  return result;
}

}  // namespace

struct Operator<FlashAttnWithKvcache, Device::Type::kNvidia, 0>::Impl {
  infini_flash::Flash_fwd_params params{};
  std::size_t workspace_bytes = 0;
  std::size_t lse_offset = 0, lse_accum_offset = 0, out_accum_offset = 0;
  std::optional<double> scale;
  int64_t requested_splits;
  bool requested_causal;
  std::array<int64_t, 2> requested_window;
  bool lengths_present;
};

bool Operator<FlashAttnWithKvcache, Device::Type::kNvidia, 0>::supported_layout(
    const Tensor q, Tensor k, Tensor v, std::optional<Tensor> lengths,
    std::optional<Tensor> blocks, Tensor out) {
  if (!plain_layout(q, 4) || !plain_layout(k, 4) || !plain_layout(v, 4) ||
      !plain_layout(out, 4) || !ordinary_head_dim_bin(q.size(3)) ||
      q.device().type() != Device::Type::kNvidia)
    return false;
  // New dimensions are restricted to ordinary dense MHA, not decode/GQA,
  // length-masked or paged cache domains that require separate qualification.
  if ((q.size(3) == 72 || q.size(3) == 80) &&
      (lengths || blocks || q.size(0) != k.size(0) || q.size(2) != k.size(2)))
    return false;
  // Upstream vectorized loads/stores require 16-byte-aligned rows and heads.
  // Metadata-only descriptor views may have null data until invocation.
  for (auto tensor : {q, k, v, out}) {
    if (tensor.data() && reinterpret_cast<std::uintptr_t>(tensor.data()) % 16)
      return false;
    for (int axis = 0; axis < 3; ++axis)
      if (tensor.size(axis) > 1 && tensor.stride(axis) % 8 != 0) return false;
  }
  if ((lengths && !plain_layout(*lengths, 1)) ||
      (blocks && (!plain_layout(*blocks, 2) ||
                  blocks->size(1) >
                      (std::numeric_limits<int>::max() - 127) / k.size(1))))
    return false;
  return true;
}

void Operator<FlashAttnWithKvcache, Device::Type::kNvidia, 0>::prepare(
    const Tensor q, Tensor cache, std::optional<Tensor> lengths,
    std::optional<Tensor> blocks, std::optional<double> scale,
    bool causal, const std::vector<int64_t>& window_size,
    int64_t splits, Tensor out, int64_t logical_bound) {
  DeviceScope device(device_index_);
  int major, sms;
  INFINI_FLASH_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_index_));
  INFINI_FLASH_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device_index_));
  if (major < 8) throw std::invalid_argument("native Attention requires SM80 or newer");
  auto state = std::make_shared<Impl>();
  state->scale = scale;
  state->requested_splits = splits;
  state->requested_causal = causal;
  state->requested_window = {window_size[0], window_size[1]};
  state->lengths_present = lengths.has_value();
  auto& p = state->params;
  p.b = q.size(0); p.h = q.size(2); p.h_k = cache.size(2);
  p.d = q.size(3); p.d_rounded = ordinary_head_dim_bin(p.d);
  p.seqlen_q = q.size(1);
  p.seqlen_k = logical_bound > 0 ? logical_bound :
      (blocks ? blocks->size(1) * cache.size(1) : cache.size(1));
  p.q_batch_stride = q.stride(0); p.q_row_stride = q.stride(1); p.q_head_stride = q.stride(2);
  p.o_batch_stride = out.stride(0); p.o_row_stride = out.stride(1); p.o_head_stride = out.stride(2);
  // Preserve upstream's no-copy decode GQA transpose, including strided Q.
  // It reinterprets heads as query positions and is valid only globally.
  const bool finite_window = window_size[0] >= 0;
  if (!finite_window && p.seqlen_q == 1 && p.h > p.h_k) {
    const int groups = p.h / p.h_k;
    p.seqlen_q = groups; p.h = p.h_k;
    p.q_row_stride = q.stride(2); p.q_head_stride = q.stride(2) * groups;
    p.o_row_stride = out.stride(2); p.o_head_stride = out.stride(2) * groups;
  }
  p.h_h_k_ratio = p.h / p.h_k;
  p.k_batch_stride = cache.stride(0); p.k_row_stride = cache.stride(1); p.k_head_stride = cache.stride(2);
  p.v_batch_stride = v_cache_strides_[0]; p.v_row_stride = v_cache_strides_[1]; p.v_head_stride = v_cache_strides_[2];
  p.seqlen_q_rounded = ((p.seqlen_q + 127) / 128) * 128;
  p.seqlen_k_rounded = ((p.seqlen_k + 127) / 128) * 128;
  // Finite right-zero masking is causal via the local kernel, not the global
  // causal specialization. Keep the real single-token query geometry.
  p.is_causal = !finite_window && causal && q.size(1) != 1;
  p.window_size_left = finite_window ? static_cast<int>(window_size[0])
                                   : (p.is_causal ? p.seqlen_k : -1);
  p.window_size_right = finite_window ? 0 : (p.is_causal ? 0 : -1);
  p.is_bf16 = q.dtype() == DataType::kBFloat16;
  p.scale_softmax = scale.value_or(1.0 / std::sqrt(static_cast<double>(p.d)));
  if (!std::isfinite(p.scale_softmax)) throw std::invalid_argument("nonfinite Attention scale");
  // Preserve upstream's double-precision M_LOG2E multiplication before the
  // final float assignment rather than introducing another rounded constant.
  p.scale_softmax_log2 = p.scale_softmax * M_LOG2E;
  p.p_dropout = p.rp_dropout = 1.f; p.p_dropout_in_uint8_t = 255;
  p.scale_softmax_rp_dropout = p.scale_softmax;
  p.is_seqlens_k_cumulative = !lengths;
  p.page_block_size = blocks ? cache.size(1) : 1;
  p.block_table_batch_stride = blocks ? blocks->stride(0) : 0;
  const auto work_blocks = product({static_cast<std::size_t>(p.b),
                                    static_cast<std::size_t>(p.h),
                                    static_cast<std::size_t>((p.seqlen_q + 63) / 64)});
  if (work_blocks > std::numeric_limits<int>::max())
    throw std::overflow_error("native Attention scheduling size overflow");
  const int block_n = native_flash::split_block_n(p.d_rounded);
  p.num_splits = splits > 0 ? splits : infini_flash::num_splits_heuristic(
      static_cast<int>(work_blocks), sms * 2,
      (p.seqlen_k + block_n - 1) / block_n, 128);
  auto bytes = product({static_cast<std::size_t>(p.b),
                        static_cast<std::size_t>(p.h),
                        static_cast<std::size_t>(p.seqlen_q), sizeof(float)});
  native_flash::WorkspaceLayout layout;
  state->lse_offset = layout.append(bytes);
  if (p.num_splits > 1) {
    const auto split_bytes = product({bytes, static_cast<std::size_t>(p.num_splits)});
    state->lse_accum_offset = layout.append(split_bytes);
    state->out_accum_offset = layout.append(product({
        split_bytes, static_cast<std::size_t>(p.d_rounded)}));
  }
  state->workspace_bytes = layout.required();
  impl_ = std::move(state);
}

std::size_t Operator<FlashAttnWithKvcache, Device::Type::kNvidia, 0>::
workspace_size_in_bytes() const { return impl_->workspace_bytes; }

Operator<FlashAttnWithKvcache, Device::Type::kNvidia, 0>::~Operator() {
  impl_.reset();
}

void Operator<FlashAttnWithKvcache, Device::Type::kNvidia, 0>::execute(
    const Tensor q, Tensor k, Tensor v, std::optional<Tensor> lengths,
    std::optional<Tensor> blocks, std::optional<double> scale,
    bool causal, const std::vector<int64_t>& window_size,
    int64_t splits, Tensor out) const {
  if (q.shape() != q_shape_ || q.strides() != q_strides_ ||
      k.shape() != k_cache_shape_ || k.strides() != k_cache_strides_ ||
      v.shape() != v_cache_shape_ || v.strides() != v_cache_strides_ ||
      out.shape() != out_shape_ || out.strides() != out_strides_ ||
      (blocks ? Tensor::Shape{blocks->shape()} : Tensor::Shape{}) != block_table_shape_ ||
      (blocks ? Tensor::Strides{blocks->strides()} : Tensor::Strides{}) != block_table_strides_ ||
      scale != impl_->scale || causal != impl_->requested_causal ||
      window_size.size() != 2 ||
      window_size[0] != impl_->requested_window[0] ||
      window_size[1] != impl_->requested_window[1] ||
      splits != impl_->requested_splits || lengths.has_value() != impl_->lengths_present ||
      q.device().index() != device_index_ || q.dtype() != q_dtype_)
    throw std::invalid_argument("native Attention prepared metadata changed");
  DeviceScope device(device_index_);
  auto stream = static_cast<cudaStream_t>(stream_);
  native_flash::WorkspaceLease scratch(workspace_, workspace_size_in_bytes_,
                                       impl_->workspace_bytes, stream,
                                       device_index_);
  auto p = impl_->params;  // Dynamic addresses, never cached input values.
  auto* workspace = static_cast<std::byte*>(scratch.data());
  p.softmax_lse_ptr = workspace + impl_->lse_offset;
  if (p.num_splits > 1) {
    p.softmax_lseaccum_ptr = workspace + impl_->lse_accum_offset;
    p.oaccum_ptr = workspace + impl_->out_accum_offset;
  }
  // Upstream parameter struct uses void* even for the read-only query.
  p.q_ptr = const_cast<void*>(q.data());
  p.k_ptr = k.data(); p.v_ptr = v.data(); p.o_ptr = out.data();
  p.cu_seqlens_k = lengths ? static_cast<int*>(lengths->data()) : nullptr;
  p.block_table = blocks ? static_cast<int*>(blocks->data()) : nullptr;
  switch (p.d_rounded) {
    case 32:
      if (p.is_bf16) {
        if (p.is_causal) native_flash::launch_32_bf16_true(p, stream);
        else native_flash::launch_32_bf16_false(p, stream);
      } else {
        if (p.is_causal) native_flash::launch_32_fp16_true(p, stream);
        else native_flash::launch_32_fp16_false(p, stream);
      }
      break;
    case 64:
      if (p.is_bf16) {
        if (p.is_causal) native_flash::launch_64_bf16_true(p, stream);
        else native_flash::launch_64_bf16_false(p, stream);
      } else {
        if (p.is_causal) native_flash::launch_64_fp16_true(p, stream);
        else native_flash::launch_64_fp16_false(p, stream);
      }
      break;
    case 96:
      if (p.is_bf16) {
        if (p.is_causal) native_flash::launch_96_bf16_true(p, stream);
        else native_flash::launch_96_bf16_false(p, stream);
      } else {
        if (p.is_causal) native_flash::launch_96_fp16_true(p, stream);
        else native_flash::launch_96_fp16_false(p, stream);
      }
      break;
    case 128:
      if (p.is_bf16) {
        if (p.is_causal) native_flash::launch_128_bf16_true(p, stream);
        else native_flash::launch_128_bf16_false(p, stream);
      } else {
        if (p.is_causal) native_flash::launch_128_fp16_true(p, stream);
        else native_flash::launch_128_fp16_false(p, stream);
      }
      break;
    case 192:
      if (p.is_bf16) {
        if (p.is_causal) native_flash::launch_192_bf16_true(p, stream);
        else native_flash::launch_192_bf16_false(p, stream);
      } else {
        if (p.is_causal) native_flash::launch_192_fp16_true(p, stream);
        else native_flash::launch_192_fp16_false(p, stream);
      }
      break;
    case 256:
      if (p.is_bf16) {
        if (p.is_causal) native_flash::launch_256_bf16_true(p, stream);
        else native_flash::launch_256_bf16_false(p, stream);
      } else {
        if (p.is_causal) native_flash::launch_256_fp16_true(p, stream);
        else native_flash::launch_256_fp16_false(p, stream);
      }
      break;
    default: throw std::logic_error("unqualified native Flash head dimension");
  }
  scratch.finish();
}
}  // namespace infini::ops
