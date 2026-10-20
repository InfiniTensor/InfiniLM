#ifndef INFINI_OPS_CPU_TOP_K_TOP_P_SAMPLING_FROM_LOGITS_KERNEL_H_
#define INFINI_OPS_CPU_TOP_K_TOP_P_SAMPLING_FROM_LOGITS_KERNEL_H_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "base/top_k_top_p_sampling_from_logits.h"
#include "data_type.h"
#include "native/cpu/caster_.h"
#include "operator.h"
#include "tensor.h"

namespace infini::ops {

template <>
class Operator<TopKTopPSamplingFromLogits, Device::Type::kCpu>
    : public TopKTopPSamplingFromLogits, Caster<Device::Type::kCpu> {
 public:
  Operator(const Tensor logits, const Tensor top_k, const Tensor top_p,
           const std::optional<Tensor> indices,
           const std::string filter_apply_order, const bool deterministic,
           const bool check_nan, const std::optional<int64_t> seed,
           const std::optional<int64_t> offset, Tensor out)
      : TopKTopPSamplingFromLogits(logits, top_k, top_p, indices,
                                   filter_apply_order, deterministic, check_nan,
                                   seed, offset, out) {
    ValidateSupportedOptions(indices, filter_apply_order, deterministic,
                             check_nan);
  }

  void operator()(const Tensor logits, const Tensor top_k, const Tensor top_p,
                  const std::optional<Tensor> indices,
                  const std::string filter_apply_order,
                  const bool deterministic, const bool check_nan,
                  const std::optional<int64_t> seed,
                  const std::optional<int64_t> offset,
                  Tensor out) const override {
    ValidateSupportedOptions(indices, filter_apply_order, deterministic,
                             check_nan);
    const auto actual_seed = static_cast<uint64_t>(
        seed.value_or(static_cast<int64_t>(std::random_device{}())));
    const auto actual_offset = static_cast<uint64_t>(offset.value_or(0));

    DispatchFunc<Device::Type::kCpu, AllFloatTypes>(
        logits.dtype(),
        [&](auto tag) {
          using T = typename decltype(tag)::type;
          Compute<T>(logits, top_k, top_p, actual_seed, actual_offset, out);
        },
        "`Operator<TopKTopPSamplingFromLogits, "
        "Device::Type::kCpu>::operator()`");
  }

 private:
  static void ValidateSupportedOptions(const std::optional<Tensor> indices,
                                       const std::string& filter_apply_order,
                                       const bool deterministic,
                                       const bool check_nan) {
    assert(!indices.has_value() &&
           "The CPU `TopKTopPSamplingFromLogits` provider does not support "
           "`indices`.");
    assert(filter_apply_order == "top_k_first" &&
           "The CPU `TopKTopPSamplingFromLogits` provider supports only "
           "`top_k_first`.");
    assert(deterministic &&
           "The CPU `TopKTopPSamplingFromLogits` provider supports only the "
           "deterministic path.");
    assert(!check_nan &&
           "The CPU `TopKTopPSamplingFromLogits` provider does not support "
           "`check_nan`.");
  }

  template <typename T>
  void Compute(const Tensor logits, const Tensor top_k, const Tensor top_p,
               uint64_t seed, uint64_t offset, Tensor out) const {
    const auto* logits_ptr = static_cast<const T*>(logits.data());
    auto* out_ptr = static_cast<int32_t*>(out.data());

    for (Tensor::Size row = 0; row < batch_size_; ++row) {
      out_ptr[row * out.stride(0)] =
          SampleRow(logits_ptr + row * logits.stride(0), logits.stride(1),
                    GetK(top_k, row), GetP(top_p, row), seed, offset + row);
    }
  }

  template <typename T>
  int32_t SampleRow(const T* row, Tensor::Stride stride, int64_t top_k,
                    double top_p, uint64_t seed, uint64_t offset) const {
    std::vector<int64_t> indices(vocab_size_);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int64_t a, int64_t b) {
      return Cast<float>(row[a * stride]) > Cast<float>(row[b * stride]);
    });

    Tensor::Size keep_count = NormalizeTopK(top_k);
    if (top_p > 0.0 && top_p < 1.0) {
      keep_count = ApplyTopP(row, stride, top_p, indices, keep_count);
    }

    if (keep_count == 1) {
      return static_cast<int32_t>(indices[0]);
    }

    std::vector<double> weights(keep_count);
    float max_val = -std::numeric_limits<float>::infinity();
    for (Tensor::Size i = 0; i < keep_count; ++i) {
      const auto value = Cast<float>(row[indices[i] * stride]);
      if (value > max_val) max_val = value;
    }

    for (Tensor::Size i = 0; i < keep_count; ++i) {
      weights[i] = std::exp(Cast<float>(row[indices[i] * stride]) - max_val);
    }

    std::discrete_distribution<Tensor::Size> dist(weights.begin(),
                                                  weights.end());
    std::mt19937_64 rng(seed);
    rng.discard(offset);
    return static_cast<int32_t>(indices[dist(rng)]);
  }

  template <typename T>
  Tensor::Size ApplyTopP(const T* row, Tensor::Stride stride, double top_p,
                         const std::vector<int64_t>& indices,
                         Tensor::Size keep_count) const {
    float max_val = -std::numeric_limits<float>::infinity();
    for (Tensor::Size i = 0; i < keep_count; ++i) {
      const auto value = Cast<float>(row[indices[i] * stride]);
      if (value > max_val) max_val = value;
    }

    double sum = 0.0;
    std::vector<double> probs(keep_count);
    for (Tensor::Size i = 0; i < keep_count; ++i) {
      probs[i] = std::exp(Cast<float>(row[indices[i] * stride]) - max_val);
      sum += probs[i];
    }

    double cumulative = 0.0;
    for (Tensor::Size i = 0; i < keep_count; ++i) {
      cumulative += probs[i] / sum;
      if (cumulative >= top_p) return i + 1;
    }

    return keep_count;
  }

  Tensor::Size NormalizeTopK(int64_t top_k) const {
    if (top_k <= 0 || static_cast<Tensor::Size>(top_k) > vocab_size_) {
      return vocab_size_;
    }
    return static_cast<Tensor::Size>(top_k);
  }

  int64_t GetK(const Tensor top_k, Tensor::Size row) const {
    const auto offset = row * top_k.stride(0);
    if (top_k.dtype() == DataType::kInt32) {
      return static_cast<const int32_t*>(top_k.data())[offset];
    }
    return static_cast<const int64_t*>(top_k.data())[offset];
  }

  double GetP(const Tensor top_p, Tensor::Size row) const {
    const auto offset = row * top_p.stride(0);
    switch (top_p.dtype()) {
      case DataType::kFloat16:
        return Cast<float>(static_cast<const Float16*>(top_p.data())[offset]);
      case DataType::kBFloat16:
        return Cast<float>(static_cast<const BFloat16*>(top_p.data())[offset]);
      case DataType::kFloat32:
        return static_cast<const float*>(top_p.data())[offset];
      case DataType::kFloat64:
        return static_cast<const double*>(top_p.data())[offset];
      default:
        assert(false &&
               "`TopKTopPSamplingFromLogits` received unsupported `top_p` "
               "dtype.");
        return 1.0;
    }
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_CPU_TOP_K_TOP_P_SAMPLING_FROM_LOGITS_KERNEL_H_
