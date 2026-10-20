#ifndef INFINI_OPS_BASE_FLASH_ATTN_WITH_KVCACHE_H_
#define INFINI_OPS_BASE_FLASH_ATTN_WITH_KVCACHE_H_

#include <cassert>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "operator.h"

namespace infini::ops {

class FlashAttnWithKvcacheConfig : public Config {
 public:
  explicit FlashAttnWithKvcacheConfig(int64_t logical_kv_upper_bound = 0)
      : logical_kv_upper_bound_{logical_kv_upper_bound} {
    if (logical_kv_upper_bound < 0) {
      throw std::invalid_argument("negative logical KV upper bound");
    }
  }

  std::unique_ptr<Config> Clone() const override {
    return std::make_unique<FlashAttnWithKvcacheConfig>(*this);
  }

  bool requires_config_aware_provider() const override {
    return logical_kv_upper_bound_ != 0;
  }

  int64_t logical_kv_upper_bound() const { return logical_kv_upper_bound_; }

  static int64_t BoundFrom(const Config& config) {
    const auto* typed = dynamic_cast<const FlashAttnWithKvcacheConfig*>(&config);
    return typed ? typed->logical_kv_upper_bound() : 0;
  }

 private:
  int64_t logical_kv_upper_bound_;
};

// Inference attention with an optional in-place KV-cache update, aligned with
// Dao-AILab FlashAttention's `flash_attn_with_kvcache` public interface.
class FlashAttnWithKvcache : public Operator<FlashAttnWithKvcache> {
 public:
  static constexpr bool kAutomaticDefault = true;

  // Common metadata contract only; each provider retains its own qualification.
  // Null storage is legal for planning, but never for a nonempty invocation.
  template <typename Lengths>
  static bool supports(
      const Tensor q, Tensor k_cache, Tensor v_cache,
      const std::optional<Tensor> k, const std::optional<Tensor> v,
      const std::optional<Tensor> rotary_cos,
      const std::optional<Tensor> rotary_sin, const Lengths cache_seqlens,
      const std::optional<Tensor> cache_batch_idx,
      const std::optional<Tensor> cache_leftpad,
      const std::optional<Tensor> block_table,
      const std::optional<Tensor> alibi_slopes,
      const std::optional<double> softmax_scale, const bool causal,
      const std::vector<int64_t> window_size, const double softcap,
      const bool rotary_interleaved, const int64_t num_splits,
      const bool return_softmax_lse, Tensor out,
      std::optional<Tensor> softmax_lse) {
    const auto lengths = [&] {
      if constexpr (std::is_convertible_v<Lengths, int64_t>)
        return std::optional<Tensor>{};
      else
        return std::optional<Tensor>{cache_seqlens};
    }();
    if constexpr (std::is_convertible_v<Lengths, int64_t>)
      if (static_cast<int64_t>(cache_seqlens) < 0) return false;
    const auto metadata = [&](const Tensor& tensor) {
      return tensor.device().type() == q.device().type() &&
             tensor.device().index() == q.device().index();
    };
    for (const auto& tensor : {q, k_cache, v_cache, out})
      if (!metadata(tensor)) return false;
    for (const auto& tensor : {k, v, rotary_cos, rotary_sin, lengths,
                              cache_batch_idx, cache_leftpad, block_table,
                              alibi_slopes, softmax_lse})
      if (tensor && !metadata(*tensor)) return false;
    if (q.ndim() != 4 || k_cache.ndim() != 4 || v_cache.ndim() != 4 ||
        out.shape() != q.shape() || k_cache.shape() != v_cache.shape() ||
        (q.dtype() != DataType::kFloat16 && q.dtype() != DataType::kBFloat16) ||
        k_cache.dtype() != q.dtype() || v_cache.dtype() != q.dtype() ||
        out.dtype() != q.dtype() || q.size(0) == 0 || q.size(1) == 0 ||
        q.size(2) == 0 || k_cache.size(0) == 0 || k_cache.size(1) == 0 ||
        k_cache.size(2) == 0 || q.size(2) % k_cache.size(2) != 0 ||
        q.size(3) == 0 || q.size(3) > 256 || q.size(3) != k_cache.size(3) ||
        q.stride(-1) != 1 || k_cache.stride(-1) != 1 ||
        v_cache.stride(-1) != 1 || out.stride(-1) != 1 ||
        k.has_value() != v.has_value() ||
        rotary_cos.has_value() != rotary_sin.has_value() ||
        return_softmax_lse != softmax_lse.has_value() ||
        window_size.size() != 2 || window_size[0] < -1 || window_size[1] < -1 ||
        !(softcap >= 0.0) || num_splits < 0) return false;
    if (k && (k->ndim() != 4 || k->shape() != v->shape() ||
              k->size(0) != q.size(0) || k->size(2) != k_cache.size(2) ||
              k->size(3) != q.size(3) || k->dtype() != q.dtype() ||
              v->dtype() != q.dtype() || k->stride(-1) != 1 ||
              v->stride(-1) != 1)) return false;
    if (rotary_cos &&
        (!k || rotary_cos->ndim() != 2 ||
         rotary_cos->shape() != rotary_sin->shape() ||
         rotary_cos->dtype() != q.dtype() || rotary_sin->dtype() != q.dtype() ||
         rotary_cos->size(1) == 0 || rotary_cos->size(1) > q.size(3) / 2 ||
         rotary_cos->size(1) % 8 != 0)) return false;
    const auto index_vector = [&](const std::optional<Tensor>& tensor) {
      return !tensor || (tensor->ndim() == 1 && tensor->size(0) == q.size(0) &&
                         tensor->dtype() == DataType::kInt32 &&
                         tensor->IsContiguous());
    };
    if (!index_vector(lengths) || !index_vector(cache_batch_idx) ||
        !index_vector(cache_leftpad)) return false;
    if (block_table) {
      if (block_table->ndim() != 2 || block_table->size(0) != q.size(0) ||
          block_table->dtype() != DataType::kInt32 ||
          !block_table->IsContiguous() || k_cache.size(1) % 256 != 0) return false;
    } else if (!cache_batch_idx && k_cache.size(0) < q.size(0)) return false;
    if (alibi_slopes &&
        (alibi_slopes->dtype() != DataType::kFloat32 ||
         !alibi_slopes->IsContiguous() ||
         !((alibi_slopes->ndim() == 1 && alibi_slopes->size(0) == q.size(2)) ||
           (alibi_slopes->ndim() == 2 && alibi_slopes->size(0) == q.size(0) &&
            alibi_slopes->size(1) == q.size(2))))) return false;
    if (softmax_lse &&
        (softmax_lse->dtype() != DataType::kFloat32 ||
         softmax_lse->shape() != Tensor::Shape{q.size(0), q.size(2), q.size(1)}))
      return false;
    (void)softmax_scale;
    (void)causal;
    (void)rotary_interleaved;
    return true;
  }

  static bool supports(const Tensor q, Tensor k_cache, Tensor v_cache, Tensor out) {
    return supports(q, k_cache, v_cache, std::nullopt, std::nullopt,
                    std::nullopt, std::nullopt, std::optional<Tensor>{},
                    std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                    std::nullopt, false, {-1, -1}, 0.0, true, 0, false, out,
                    std::nullopt);
  }

  FlashAttnWithKvcache(const Tensor q, Tensor k_cache, Tensor v_cache,
                       Tensor out)
      : FlashAttnWithKvcache{q,
                             k_cache,
                             v_cache,
                             std::nullopt,
                             std::nullopt,
                             std::nullopt,
                             std::nullopt,
                             std::optional<Tensor>{},
                             std::nullopt,
                             std::nullopt,
                             std::nullopt,
                             std::nullopt,
                             std::nullopt,
                             false,
                             {-1, -1},
                             0.0,
                             true,
                             0,
                             false,
                             out,
                             std::nullopt} {}

  FlashAttnWithKvcache(const Tensor q, Tensor k_cache, Tensor v_cache,
                       const std::optional<Tensor> k,
                       const std::optional<Tensor> v,
                       const std::optional<Tensor> rotary_cos,
                       const std::optional<Tensor> rotary_sin,
                       const int64_t cache_seqlens,
                       const std::optional<Tensor> cache_batch_idx,
                       const std::optional<Tensor> cache_leftpad,
                       const std::optional<Tensor> block_table,
                       const std::optional<Tensor> alibi_slopes,
                       const std::optional<double> softmax_scale,
                       const bool causal,
                       const std::vector<int64_t> window_size,
                       const double softcap, const bool rotary_interleaved,
                       const int64_t num_splits, const bool return_softmax_lse,
                       Tensor out, std::optional<Tensor> softmax_lse)
      : FlashAttnWithKvcache{q,
                             k_cache,
                             v_cache,
                             k,
                             v,
                             rotary_cos,
                             rotary_sin,
                             std::optional<Tensor>{},
                             cache_batch_idx,
                             cache_leftpad,
                             block_table,
                             alibi_slopes,
                             softmax_scale,
                             causal,
                             window_size,
                             softcap,
                             rotary_interleaved,
                             num_splits,
                             return_softmax_lse,
                             out,
                             softmax_lse} {
    assert(cache_seqlens >= 0 &&
           "`FlashAttnWithKvcache` requires non-negative scalar "
           "`cache_seqlens`");
  }

  FlashAttnWithKvcache(const Tensor q, Tensor k_cache, Tensor v_cache,
                       const std::optional<Tensor> k,
                       const std::optional<Tensor> v,
                       const std::optional<Tensor> rotary_cos,
                       const std::optional<Tensor> rotary_sin,
                       const std::optional<Tensor> cache_seqlens,
                       const std::optional<Tensor> cache_batch_idx,
                       const std::optional<Tensor> cache_leftpad,
                       const std::optional<Tensor> block_table,
                       const std::optional<Tensor> alibi_slopes,
                       const std::optional<double> softmax_scale,
                       const bool causal,
                       const std::vector<int64_t> window_size,
                       const double softcap, const bool rotary_interleaved,
                       const int64_t num_splits, const bool return_softmax_lse,
                       Tensor out, std::optional<Tensor> softmax_lse)
      : q_shape_{q.shape()},
        k_cache_shape_{k_cache.shape()},
        v_cache_shape_{v_cache.shape()},
        k_shape_{k.has_value() ? Tensor::Shape{k->shape()} : Tensor::Shape{}},
        v_shape_{v.has_value() ? Tensor::Shape{v->shape()} : Tensor::Shape{}},
        rotary_cos_shape_{rotary_cos.has_value()
                              ? Tensor::Shape{rotary_cos->shape()}
                              : Tensor::Shape{}},
        rotary_sin_shape_{rotary_sin.has_value()
                              ? Tensor::Shape{rotary_sin->shape()}
                              : Tensor::Shape{}},
        cache_seqlens_shape_{cache_seqlens.has_value()
                                 ? Tensor::Shape{cache_seqlens->shape()}
                                 : Tensor::Shape{}},
        cache_batch_idx_shape_{cache_batch_idx.has_value()
                                   ? Tensor::Shape{cache_batch_idx->shape()}
                                   : Tensor::Shape{}},
        cache_leftpad_shape_{cache_leftpad.has_value()
                                 ? Tensor::Shape{cache_leftpad->shape()}
                                 : Tensor::Shape{}},
        block_table_shape_{block_table.has_value()
                               ? Tensor::Shape{block_table->shape()}
                               : Tensor::Shape{}},
        alibi_slopes_shape_{alibi_slopes.has_value()
                                ? Tensor::Shape{alibi_slopes->shape()}
                                : Tensor::Shape{}},
        out_shape_{out.shape()},
        softmax_lse_shape_{softmax_lse.has_value()
                               ? Tensor::Shape{softmax_lse->shape()}
                               : Tensor::Shape{}},
        q_strides_{q.strides()},
        k_cache_strides_{k_cache.strides()},
        v_cache_strides_{v_cache.strides()},
        k_strides_{k.has_value() ? Tensor::Strides{k->strides()}
                                 : Tensor::Strides{}},
        v_strides_{v.has_value() ? Tensor::Strides{v->strides()}
                                 : Tensor::Strides{}},
        rotary_cos_strides_{rotary_cos.has_value()
                                ? Tensor::Strides{rotary_cos->strides()}
                                : Tensor::Strides{}},
        rotary_sin_strides_{rotary_sin.has_value()
                                ? Tensor::Strides{rotary_sin->strides()}
                                : Tensor::Strides{}},
        cache_seqlens_strides_{cache_seqlens.has_value()
                                   ? Tensor::Strides{cache_seqlens->strides()}
                                   : Tensor::Strides{}},
        cache_batch_idx_strides_{
            cache_batch_idx.has_value()
                ? Tensor::Strides{cache_batch_idx->strides()}
                : Tensor::Strides{}},
        cache_leftpad_strides_{cache_leftpad.has_value()
                                   ? Tensor::Strides{cache_leftpad->strides()}
                                   : Tensor::Strides{}},
        block_table_strides_{block_table.has_value()
                                 ? Tensor::Strides{block_table->strides()}
                                 : Tensor::Strides{}},
        alibi_slopes_strides_{alibi_slopes.has_value()
                                  ? Tensor::Strides{alibi_slopes->strides()}
                                  : Tensor::Strides{}},
        out_strides_{out.strides()},
        softmax_lse_strides_{softmax_lse.has_value()
                                 ? Tensor::Strides{softmax_lse->strides()}
                                 : Tensor::Strides{}},
        q_dtype_{q.dtype()},
        k_cache_dtype_{k_cache.dtype()},
        v_cache_dtype_{v_cache.dtype()},
        k_dtype_{k.has_value() ? k->dtype() : q.dtype()},
        v_dtype_{v.has_value() ? v->dtype() : q.dtype()},
        rotary_cos_dtype_{rotary_cos.has_value() ? rotary_cos->dtype()
                                                 : q.dtype()},
        rotary_sin_dtype_{rotary_sin.has_value() ? rotary_sin->dtype()
                                                 : q.dtype()},
        cache_seqlens_dtype_{cache_seqlens.has_value() ? cache_seqlens->dtype()
                                                       : DataType::kInt32},
        cache_batch_idx_dtype_{cache_batch_idx.has_value()
                                   ? cache_batch_idx->dtype()
                                   : DataType::kInt32},
        cache_leftpad_dtype_{cache_leftpad.has_value() ? cache_leftpad->dtype()
                                                       : DataType::kInt32},
        block_table_dtype_{block_table.has_value() ? block_table->dtype()
                                                   : DataType::kInt32},
        alibi_slopes_dtype_{alibi_slopes.has_value() ? alibi_slopes->dtype()
                                                     : DataType::kFloat32},
        out_dtype_{out.dtype()},
        softmax_lse_dtype_{softmax_lse.has_value() ? softmax_lse->dtype()
                                                   : DataType::kFloat32},
        has_k_{k.has_value()},
        has_v_{v.has_value()},
        has_rotary_cos_{rotary_cos.has_value()},
        has_rotary_sin_{rotary_sin.has_value()},
        has_cache_batch_idx_{cache_batch_idx.has_value()},
        has_block_table_{block_table.has_value()},
        has_alibi_slopes_{alibi_slopes.has_value()},
        has_softmax_lse_{softmax_lse.has_value()},
        batch_size_{q.ndim() > 0 ? q.size(0) : 0},
        head_size_{q.ndim() == 4 ? q.size(3) : 0},
        device_index_{q.device().index()} {
    assert(q.ndim() == 4 && k_cache.ndim() == 4 && v_cache.ndim() == 4 &&
           "`FlashAttnWithKvcache` requires 4D `q`, `k_cache`, and "
           "`v_cache`");
    assert(k_cache.shape() == v_cache.shape() &&
           "`FlashAttnWithKvcache` requires matching K/V cache shapes");
    assert(
        (q_dtype_ == DataType::kFloat16 || q_dtype_ == DataType::kBFloat16) &&
        q_dtype_ == k_cache_dtype_ && q_dtype_ == v_cache_dtype_ &&
        q_dtype_ == out_dtype_ &&
        "`FlashAttnWithKvcache` requires matching float16 or bfloat16 "
        "Q, cache, and output dtypes");
    assert(q.size(0) > 0 && q.size(1) > 0 && q.size(2) > 0 &&
           k_cache.size(0) > 0 && k_cache.size(1) > 0 && k_cache.size(2) > 0 &&
           "`FlashAttnWithKvcache` requires non-empty Q and KV cache "
           "dimensions");
    assert(q.size(2) % k_cache.size(2) == 0 && q.size(3) == k_cache.size(3) &&
           "`FlashAttnWithKvcache` requires compatible Q and KV heads");
    assert(head_size_ > 0 && head_size_ <= 256 &&
           "`FlashAttnWithKvcache` requires a head dimension no greater than "
           "256");
    assert(out.shape() == q.shape() &&
           "`FlashAttnWithKvcache` output must have the same shape as Q");
    assert(return_softmax_lse == has_softmax_lse_ &&
           "`FlashAttnWithKvcache` requires `softmax_lse` exactly when "
           "`return_softmax_lse` is true");
    if (has_softmax_lse_) {
      assert((softmax_lse->shape() ==
              Tensor::Shape{q.size(0), q.size(2), q.size(1)}) &&
             softmax_lse_dtype_ == DataType::kFloat32 &&
             "`FlashAttnWithKvcache` softmax LSE output must have shape "
             "(batch_size, num_heads, seqlen) and dtype float32");
    }
    assert(q.stride(-1) == 1 && k_cache.stride(-1) == 1 &&
           v_cache.stride(-1) == 1 && out.stride(-1) == 1 &&
           "`FlashAttnWithKvcache` requires contiguous head dimensions");
    assert(has_k_ == has_v_ &&
           "`FlashAttnWithKvcache` requires `k` and `v` together");
    if (has_k_) {
      assert(k->ndim() == 4 && k->shape() == v->shape() &&
             k->size(0) == batch_size_ && k->size(2) == k_cache.size(2) &&
             k->size(3) == head_size_ &&
             "`FlashAttnWithKvcache` received incompatible new K/V shapes");
      assert(k_dtype_ == q_dtype_ && v_dtype_ == q_dtype_ &&
             "`FlashAttnWithKvcache` requires matching new K/V dtypes");
      assert(k->stride(-1) == 1 && v->stride(-1) == 1 &&
             "`FlashAttnWithKvcache` requires contiguous new K/V head "
             "dimensions");
    }
    assert(has_rotary_cos_ == has_rotary_sin_ &&
           "`FlashAttnWithKvcache` requires rotary cosine and sine together");
    if (has_rotary_cos_) {
      assert(has_k_ && rotary_cos->ndim() == 2 &&
             rotary_cos->shape() == rotary_sin->shape() &&
             rotary_cos_dtype_ == q_dtype_ && rotary_sin_dtype_ == q_dtype_ &&
             rotary_cos->size(1) > 0 && rotary_cos->size(1) * 2 <= head_size_ &&
             (rotary_cos->size(1) * 2) % 16 == 0 &&
             "`FlashAttnWithKvcache` received incompatible rotary tables");
    }
    ValidateIndexVector(cache_seqlens);
    ValidateIndexVector(cache_batch_idx);
    ValidateIndexVector(cache_leftpad);
    if (has_block_table_) {
      assert(block_table->ndim() == 2 && block_table->size(0) == batch_size_ &&
             block_table_dtype_ == DataType::kInt32 &&
             block_table->IsContiguous() && k_cache.size(1) % 256 == 0 &&
             "`FlashAttnWithKvcache` requires a contiguous int32 block table "
             "and page size divisible by 256");
    } else {
      assert((has_cache_batch_idx_ || k_cache.size(0) >= batch_size_) &&
             "`FlashAttnWithKvcache` cache batch is too small");
    }
    if (has_alibi_slopes_) {
      assert((alibi_slopes->ndim() == 1 || alibi_slopes->ndim() == 2) &&
             alibi_slopes_dtype_ == DataType::kFloat32 &&
             alibi_slopes->IsContiguous() &&
             "`FlashAttnWithKvcache` requires contiguous float32 ALiBi "
             "slopes");
      assert(
          ((alibi_slopes->ndim() == 1 && alibi_slopes->size(0) == q.size(2)) ||
           (alibi_slopes->ndim() == 2 && alibi_slopes->size(0) == batch_size_ &&
            alibi_slopes->size(1) == q.size(2))) &&
          "`FlashAttnWithKvcache` received incompatible ALiBi slopes");
    }
    assert(window_size.size() == 2 && window_size[0] >= -1 &&
           window_size[1] >= -1 &&
           "`FlashAttnWithKvcache` `window_size` must contain two values >= "
           "-1");
    assert(softcap >= 0.0 &&
           "`FlashAttnWithKvcache` requires non-negative `softcap`");
    assert(num_splits >= 0 &&
           "`FlashAttnWithKvcache` requires non-negative `num_splits`");
    const auto same_device_as_q = [&](const Tensor tensor) {
      return tensor.device().type() == q.device().type() &&
             tensor.device().index() == q.device().index();
    };
    assert(
        same_device_as_q(k_cache) && same_device_as_q(v_cache) &&
        same_device_as_q(out) &&
        (!softmax_lse.has_value() || same_device_as_q(*softmax_lse)) &&
        (!k.has_value() || same_device_as_q(*k)) &&
        (!v.has_value() || same_device_as_q(*v)) &&
        (!rotary_cos.has_value() || same_device_as_q(*rotary_cos)) &&
        (!rotary_sin.has_value() || same_device_as_q(*rotary_sin)) &&
        (!cache_seqlens.has_value() || same_device_as_q(*cache_seqlens)) &&
        (!cache_batch_idx.has_value() || same_device_as_q(*cache_batch_idx)) &&
        (!cache_leftpad.has_value() || same_device_as_q(*cache_leftpad)) &&
        (!block_table.has_value() || same_device_as_q(*block_table)) &&
        (!alibi_slopes.has_value() || same_device_as_q(*alibi_slopes)) &&
        "`FlashAttnWithKvcache` tensors must be on the same device");

    (void)softmax_scale;
    (void)causal;
    (void)rotary_interleaved;
  }

  void operator()(const Tensor q, Tensor k_cache, Tensor v_cache,
                  Tensor out) const {
    (*this)(q, k_cache, v_cache, std::nullopt, std::nullopt, std::nullopt,
            std::nullopt, std::optional<Tensor>{}, std::nullopt, std::nullopt,
            std::nullopt, std::nullopt, std::nullopt, false, {-1, -1}, 0.0,
            true, 0, false, out, std::nullopt);
  }

  virtual void operator()(
      const Tensor q, Tensor k_cache, Tensor v_cache,
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
      std::optional<Tensor> softmax_lse) const = 0;

  virtual void operator()(const Tensor q, Tensor k_cache, Tensor v_cache,
                          const std::optional<Tensor> k,
                          const std::optional<Tensor> v,
                          const std::optional<Tensor> rotary_cos,
                          const std::optional<Tensor> rotary_sin,
                          const std::optional<Tensor> cache_seqlens,
                          const std::optional<Tensor> cache_batch_idx,
                          const std::optional<Tensor> cache_leftpad,
                          const std::optional<Tensor> block_table,
                          const std::optional<Tensor> alibi_slopes,
                          const std::optional<double> softmax_scale,
                          const bool causal,
                          const std::vector<int64_t> window_size,
                          const double softcap, const bool rotary_interleaved,
                          const int64_t num_splits,
                          const bool return_softmax_lse, Tensor out,
                          std::optional<Tensor> softmax_lse) const = 0;

 protected:
  void ValidateIndexVector(const std::optional<Tensor>& tensor) const {
    if (!tensor.has_value()) {
      return;
    }
    assert(tensor->ndim() == 1 && tensor->size(0) == batch_size_ &&
           tensor->dtype() == DataType::kInt32 && tensor->IsContiguous() &&
           "`FlashAttnWithKvcache` index metadata must be contiguous int32 "
           "vectors with one value per query batch");
  }

  Tensor::Shape q_shape_;

  Tensor::Shape k_cache_shape_;

  Tensor::Shape v_cache_shape_;

  Tensor::Shape k_shape_;

  Tensor::Shape v_shape_;

  Tensor::Shape rotary_cos_shape_;

  Tensor::Shape rotary_sin_shape_;

  Tensor::Shape cache_seqlens_shape_;

  Tensor::Shape cache_batch_idx_shape_;

  Tensor::Shape cache_leftpad_shape_;

  Tensor::Shape block_table_shape_;

  Tensor::Shape alibi_slopes_shape_;

  Tensor::Shape out_shape_;

  Tensor::Shape softmax_lse_shape_;

  Tensor::Strides q_strides_;

  Tensor::Strides k_cache_strides_;

  Tensor::Strides v_cache_strides_;

  Tensor::Strides k_strides_;

  Tensor::Strides v_strides_;

  Tensor::Strides rotary_cos_strides_;

  Tensor::Strides rotary_sin_strides_;

  Tensor::Strides cache_seqlens_strides_;

  Tensor::Strides cache_batch_idx_strides_;

  Tensor::Strides cache_leftpad_strides_;

  Tensor::Strides block_table_strides_;

  Tensor::Strides alibi_slopes_strides_;

  Tensor::Strides out_strides_;

  Tensor::Strides softmax_lse_strides_;

  DataType q_dtype_;

  DataType k_cache_dtype_;

  DataType v_cache_dtype_;

  DataType k_dtype_;

  DataType v_dtype_;

  DataType rotary_cos_dtype_;

  DataType rotary_sin_dtype_;

  DataType cache_seqlens_dtype_;

  DataType cache_batch_idx_dtype_;

  DataType cache_leftpad_dtype_;

  DataType block_table_dtype_;

  DataType alibi_slopes_dtype_;

  DataType out_dtype_;

  DataType softmax_lse_dtype_;

  bool has_k_{false};

  bool has_v_{false};

  bool has_rotary_cos_{false};

  bool has_rotary_sin_{false};

  bool has_cache_batch_idx_{false};

  bool has_block_table_{false};

  bool has_alibi_slopes_{false};

  bool has_softmax_lse_{false};

  Tensor::Size batch_size_{0};

  Tensor::Size head_size_{0};

  int device_index_{0};
};

template <>
struct CacheKeyBuilder<FlashAttnWithKvcache> {
  template <typename... Args>
  detail::CacheKey operator()(const Config& config, const Args&... args) const {
    const auto bound = FlashAttnWithKvcacheConfig::BoundFrom(config);
    if (bound == 0) {
      return detail::CacheKey::Build(config.implementation_index(), args...);
    }
    return detail::CacheKey::Build(config.implementation_index(), bound,
                                   args...);
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_FLASH_ATTN_WITH_KVCACHE_H_
