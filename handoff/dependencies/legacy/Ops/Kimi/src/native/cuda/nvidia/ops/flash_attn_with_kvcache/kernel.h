#pragma once

#include <memory>
#include <limits>
#include <type_traits>
#include "base/flash_attn_with_kvcache.h"

// These macros only spell the existing public overloads, not a second API.
#define INFINI_NATIVE_KVCACHE_PARAMS(Lengths)                              \
  const Tensor q, Tensor k_cache, Tensor v_cache,                          \
  const std::optional<Tensor> k, const std::optional<Tensor> v,            \
  const std::optional<Tensor> rotary_cos,                                 \
  const std::optional<Tensor> rotary_sin, const Lengths cache_seqlens,     \
  const std::optional<Tensor> cache_batch_idx,                             \
  const std::optional<Tensor> cache_leftpad,                              \
  const std::optional<Tensor> block_table,                                \
  const std::optional<Tensor> alibi_slopes,                               \
  const std::optional<double> softmax_scale, const bool causal,           \
  const std::vector<int64_t> window_size, const double softcap,           \
  const bool rotary_interleaved, const int64_t num_splits,                \
  const bool return_softmax_lse, Tensor out,                             \
  std::optional<Tensor> softmax_lse
#define INFINI_NATIVE_KVCACHE_VALUES                                     \
  q, k_cache, v_cache, k, v, rotary_cos, rotary_sin, cache_seqlens,         \
  cache_batch_idx, cache_leftpad, block_table, alibi_slopes,               \
  softmax_scale, causal, window_size, softcap, rotary_interleaved,         \
  num_splits, return_softmax_lse, out, softmax_lse

namespace infini::ops {

// Native inference provider: qualified heads, read-only cache, global or finite
// left-window/right-zero attention, no ALiBi/softcap.
// Other existing configurations remain with their previous providers. The
// explicit build option is not enabled in the accepted LM candidate yet.
template <>
class Operator<FlashAttnWithKvcache, Device::Type::kNvidia, 0>
    : public FlashAttnWithKvcache {
 public:
  template <typename Lengths>
  static bool supports(INFINI_NATIVE_KVCACHE_PARAMS(Lengths)) {
    if constexpr (!std::is_same_v<Lengths, std::optional<Tensor>>) {
      return false;  // Scalar metadata needs a separate prepared device buffer.
    } else {
      return FlashAttnWithKvcache::supports(INFINI_NATIVE_KVCACHE_VALUES) &&
             !k && !v && !rotary_cos && !rotary_sin && !cache_batch_idx &&
             !cache_leftpad && !alibi_slopes &&
             ((window_size[0] == -1 && window_size[1] == -1) ||
              (window_size[0] >= 0 &&
               window_size[0] <= std::numeric_limits<int>::max() - 127 &&
               window_size[1] == 0)) &&
             softcap == 0.0 && num_splits <= 128 && !return_softmax_lse &&
             supported_layout(q, k_cache, v_cache, cache_seqlens, block_table, out) &&
             ((q.size(3) != 72 && q.size(3) != 80) ||
              (!causal && window_size[0] == -1));
    }
  }

  static bool supports(const Tensor q, Tensor k_cache, Tensor v_cache, Tensor out) {
    return FlashAttnWithKvcache::supports(q, k_cache, v_cache, out) &&
           supported_layout(q, k_cache, v_cache, {}, {}, out);
  }

  template <typename Lengths>
  static bool supports(const Config& config,
                       INFINI_NATIVE_KVCACHE_PARAMS(Lengths)) {
    if constexpr (!std::is_same_v<Lengths, std::optional<Tensor>>) {
      return false;
    } else {
      if (!supports(INFINI_NATIVE_KVCACHE_VALUES)) return false;
      const auto bound = FlashAttnWithKvcacheConfig::BoundFrom(config);
      if (bound == 0) return true;
      const auto capacity = block_table
          ? block_table->size(1) * k_cache.size(1) : k_cache.size(1);
      return bound <= std::numeric_limits<int>::max() - 127 &&
             static_cast<std::uint64_t>(bound) <= capacity &&
             (cache_seqlens.has_value() ||
              static_cast<std::uint64_t>(bound) == capacity);
    }
  }

  static bool supports(const Config& config, const Tensor q, Tensor k_cache,
                       Tensor v_cache, Tensor out) {
    const auto bound = FlashAttnWithKvcacheConfig::BoundFrom(config);
    return supports(q, k_cache, v_cache, out) &&
           (bound == 0 ||
            (bound <= std::numeric_limits<int>::max() - 127 &&
             static_cast<std::uint64_t>(bound) == k_cache.size(1)));
  }

  template <typename Lengths>
  Operator(INFINI_NATIVE_KVCACHE_PARAMS(Lengths))
      : Operator(Config{}, INFINI_NATIVE_KVCACHE_VALUES) {}

  template <typename Lengths>
  Operator(const Config& config, INFINI_NATIVE_KVCACHE_PARAMS(Lengths))
      : FlashAttnWithKvcache{
            checked_q(config, INFINI_NATIVE_KVCACHE_VALUES), k_cache, v_cache, k, v,
            rotary_cos, rotary_sin, cache_seqlens, cache_batch_idx, cache_leftpad,
            block_table, alibi_slopes, softmax_scale, causal, window_size,
            softcap, rotary_interleaved, num_splits, return_softmax_lse, out,
            softmax_lse} {
    if constexpr (std::is_same_v<Lengths, std::optional<Tensor>>) {
      prepare(q, k_cache, cache_seqlens, block_table, softmax_scale, causal,
              window_size, num_splits, out,
              FlashAttnWithKvcacheConfig::BoundFrom(config));
    }
  }

  Operator(const Tensor q, Tensor k_cache, Tensor v_cache, Tensor out)
      : Operator(q, k_cache, v_cache, std::nullopt, std::nullopt, std::nullopt,
                 std::nullopt, std::optional<Tensor>{}, std::nullopt,
                 std::nullopt, std::nullopt, std::nullopt, std::nullopt, false,
                 {-1, -1}, 0.0, true, 0, false, out, std::nullopt) {}
  Operator(const Config& config, const Tensor q, Tensor k_cache,
           Tensor v_cache, Tensor out)
      : Operator(config, q, k_cache, v_cache, std::nullopt, std::nullopt,
                 std::nullopt, std::nullopt, std::optional<Tensor>{},
                 std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                 std::nullopt, false, {-1, -1}, 0.0, true, 0, false, out,
                 std::nullopt) {}
  ~Operator() override;
  using FlashAttnWithKvcache::operator();

  void operator()(INFINI_NATIVE_KVCACHE_PARAMS(int64_t)) const override {
    throw std::invalid_argument("native KV Attention requires device length metadata");
  }
  void operator()(INFINI_NATIVE_KVCACHE_PARAMS(std::optional<Tensor>)) const override {
    checked_q(INFINI_NATIVE_KVCACHE_VALUES);
    execute(q, k_cache, v_cache, cache_seqlens, block_table, softmax_scale,
            causal, window_size, num_splits, out);
  }

  // Graph callers must supply stable external workspace for the graph lifetime.
  // The qualified native domain is enforced by supports(), not the GPU type.
  bool capture_safe() const override {
    // New ordinary head dimensions are SOURCE_ONLY, never capture-qualified.
    return q_shape_[3] != 72 && q_shape_[3] != 80;
  }
  std::size_t workspace_size_in_bytes() const override;

 private:
  template <typename Lengths>
  static Tensor checked_q(INFINI_NATIVE_KVCACHE_PARAMS(Lengths)) {
    if (!supports(INFINI_NATIVE_KVCACHE_VALUES)) {
      throw std::invalid_argument("unsupported native KV Attention configuration");
    }
    return q;
  }
  template <typename Lengths>
  static Tensor checked_q(const Config& config,
                          INFINI_NATIVE_KVCACHE_PARAMS(Lengths)) {
    if (!supports(config, INFINI_NATIVE_KVCACHE_VALUES)) {
      throw std::invalid_argument("unsupported native KV Attention logical bound");
    }
    return q;
  }
  static bool supported_layout(const Tensor q, Tensor k_cache, Tensor v_cache,
                               std::optional<Tensor> lengths,
                               std::optional<Tensor> blocks, Tensor out);
  void prepare(const Tensor q, Tensor cache, std::optional<Tensor> lengths,
               std::optional<Tensor> blocks, std::optional<double> scale,
               bool causal, const std::vector<int64_t>& window_size,
               int64_t splits, Tensor out, int64_t logical_bound);
  void execute(const Tensor q, Tensor k_cache, Tensor v_cache,
               std::optional<Tensor> lengths, std::optional<Tensor> blocks,
               std::optional<double> scale, bool causal,
               const std::vector<int64_t>& window_size, int64_t splits, Tensor out) const;
  struct Impl;
  std::shared_ptr<Impl> impl_;  // Complete type is confined to the provider TU.
};

}  // namespace infini::ops

#undef INFINI_NATIVE_KVCACHE_PARAMS
#undef INFINI_NATIVE_KVCACHE_VALUES
