#ifndef INFINI_OPS_BASE_TOP_K_TOP_P_SAMPLING_FROM_LOGITS_H_
#define INFINI_OPS_BASE_TOP_K_TOP_P_SAMPLING_FROM_LOGITS_H_

#include <cassert>
#include <cstdint>
#include <optional>
#include <string>

#include "data_type.h"
#include "operator.h"
#include "tensor.h"

namespace infini::ops {

// Targets the public FlashInfer `top_k_top_p_sampling_from_logits` interface.
class TopKTopPSamplingFromLogits : public Operator<TopKTopPSamplingFromLogits> {
 public:
  TopKTopPSamplingFromLogits(const Tensor logits, const Tensor top_k,
                             const Tensor top_p,
                             const std::optional<Tensor> indices,
                             const std::string filter_apply_order,
                             const bool deterministic, const bool check_nan,
                             const std::optional<int64_t> seed,
                             const std::optional<int64_t> offset, Tensor out)
      : batch_size_{out.size(0)},
        vocab_size_{logits.size(1)},
        dtype_{logits.dtype()} {
    assert(logits.ndim() == 2 &&
           "`TopKTopPSamplingFromLogits` requires 2D "
           "`[batch_size, vocab_size]` logits.");
    assert(IsFloatDtype(dtype_) &&
           "`TopKTopPSamplingFromLogits` requires floating-point logits.");
    assert(top_k.ndim() == 1 && top_k.size(0) == batch_size_ &&
           IsIntegerDtype(top_k.dtype()) &&
           "`TopKTopPSamplingFromLogits` requires integer `top_k` with shape "
           "`[batch_size]`.");
    assert(top_p.ndim() == 1 && top_p.size(0) == batch_size_ &&
           IsFloatDtype(top_p.dtype()) &&
           "`TopKTopPSamplingFromLogits` requires floating-point `top_p` with "
           "shape `[batch_size]`.");
    assert(out.ndim() == 1 &&
           "`TopKTopPSamplingFromLogits` requires 1D output.");
    assert((filter_apply_order == "top_k_first" ||
            filter_apply_order == "joint") &&
           "`TopKTopPSamplingFromLogits` requires `filter_apply_order` to be "
           "`top_k_first` or `joint`.");
    assert((!offset.has_value() || *offset >= 0) &&
           "`TopKTopPSamplingFromLogits` requires a nonnegative `offset`.");

    if (indices.has_value()) {
      assert(indices->ndim() == 1 && indices->size(0) == batch_size_ &&
             IsIntegerDtype(indices->dtype()) &&
             "`TopKTopPSamplingFromLogits` requires integer `indices` with "
             "shape `[batch_size]`.");
      assert(out.dtype() == indices->dtype() &&
             "`TopKTopPSamplingFromLogits` requires output and `indices` to "
             "have the same dtype.");
    } else {
      assert(logits.size(0) == batch_size_ &&
             "`TopKTopPSamplingFromLogits` requires output batch size to "
             "match logits when `indices` is absent.");
      assert(out.dtype() == DataType::kInt32 &&
             "`TopKTopPSamplingFromLogits` requires int32 output when "
             "`indices` is absent.");
    }

    (void)deterministic;
    (void)check_nan;
    (void)seed;
  }

  virtual void operator()(const Tensor logits, const Tensor top_k,
                          const Tensor top_p,
                          const std::optional<Tensor> indices,
                          const std::string filter_apply_order,
                          const bool deterministic, const bool check_nan,
                          const std::optional<int64_t> seed,
                          const std::optional<int64_t> offset,
                          Tensor out) const = 0;

 protected:
  static bool IsFloatDtype(DataType dtype) {
    return dtype == DataType::kFloat16 || dtype == DataType::kBFloat16 ||
           dtype == DataType::kFloat32 || dtype == DataType::kFloat64;
  }

  static bool IsIntegerDtype(DataType dtype) {
    return dtype == DataType::kInt32 || dtype == DataType::kInt64;
  }

  Tensor::Size batch_size_{0};

  Tensor::Size vocab_size_{0};

  DataType dtype_;
};

template <>
struct CacheKeyBuilder<TopKTopPSamplingFromLogits> {
  detail::CacheKey operator()(const Config& config, const Tensor logits,
                              const Tensor top_k, const Tensor top_p,
                              const std::optional<Tensor> indices,
                              const std::string filter_apply_order,
                              const bool deterministic, const bool check_nan,
                              const std::optional<int64_t> /*seed*/,
                              const std::optional<int64_t> /*offset*/,
                              Tensor out) const {
    return detail::CacheKey::Build(config.implementation_index(), logits, top_k,
                                   top_p, indices, filter_apply_order,
                                   deterministic, check_nan, out);
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_TOP_K_TOP_P_SAMPLING_FROM_LOGITS_H_
