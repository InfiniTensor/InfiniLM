#ifndef INFINI_OPS_CPU_TOP_K_TOP_P_SAMPLER_H_
#define INFINI_OPS_CPU_TOP_K_TOP_P_SAMPLER_H_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <vector>

#include "base/top_k_top_p_sampler.h"
#include "data_type.h"
#include "native/cpu/caster_.h"
#include "operator.h"
#include "tensor.h"

namespace infini::ops {

template <>
class Operator<TopKTopPSampler, Device::Type::kCpu>
    : public TopKTopPSampler, Caster<Device::Type::kCpu> {
 public:
  Operator(const Tensor logits, std::optional<Tensor> k,
           std::optional<Tensor> p, Tensor out)
      : TopKTopPSampler(logits, k, p, out) {}

  void operator()(const Tensor logits, std::optional<Tensor> k,
                  std::optional<Tensor> p, Tensor out) const override {
    DispatchFunc<Device::Type::kCpu, AllFloatTypes>(
        logits.dtype(),
        [&](auto tag) {
          using T = typename decltype(tag)::type;
          Compute<T>(logits, k, p, out);
        },
        "`Operator<TopKTopPSampler, Device::Type::kCpu>::operator()`");
  }

 private:
  template <typename T>
  void Compute(const Tensor logits, std::optional<Tensor> k,
               std::optional<Tensor> p, Tensor out) const {
    const auto* logits_ptr = static_cast<const T*>(logits.data());
    auto* out_ptr = static_cast<int32_t*>(out.data());

    for (Tensor::Size row = 0; row < batch_size_; ++row) {
      const int64_t top_k = GetK(k, row);
      const double top_p = GetP(p, row);
      out_ptr[row * out.stride(0)] = SampleRow(
          logits_ptr + row * logits.stride(0), logits.stride(1), top_k, top_p);
    }
  }

  template <typename T>
  int32_t SampleRow(const T* row, Tensor::Stride stride, int64_t top_k,
                    double top_p) const {
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
      const auto v = Cast<float>(row[indices[i] * stride]);
      if (v > max_val) max_val = v;
    }

    for (Tensor::Size i = 0; i < keep_count; ++i) {
      weights[i] = std::exp(Cast<float>(row[indices[i] * stride]) - max_val);
    }

    std::discrete_distribution<Tensor::Size> dist(weights.begin(),
                                                  weights.end());
    return static_cast<int32_t>(indices[dist(rng_)]);
  }

  template <typename T>
  Tensor::Size ApplyTopP(const T* row, Tensor::Stride stride, double top_p,
                         const std::vector<int64_t>& indices,
                         Tensor::Size keep_count) const {
    float max_val = -std::numeric_limits<float>::infinity();
    for (Tensor::Size i = 0; i < keep_count; ++i) {
      const auto v = Cast<float>(row[indices[i] * stride]);
      if (v > max_val) max_val = v;
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
      if (cumulative >= top_p) {
        return i + 1;
      }
    }

    return keep_count;
  }

  Tensor::Size NormalizeTopK(int64_t top_k) const {
    if (top_k <= 0 || static_cast<Tensor::Size>(top_k) >
                          static_cast<Tensor::Size>(vocab_size_)) {
      return vocab_size_;
    }
    return static_cast<Tensor::Size>(top_k);
  }

  int64_t GetK(std::optional<Tensor> k, Tensor::Size row) const {
    if (!k.has_value()) return static_cast<int64_t>(vocab_size_);

    const auto offset = (k->size(0) == 1 ? 0 : row) * k->stride(0);
    if (k->dtype() == DataType::kInt32) {
      return static_cast<const int32_t*>(k->data())[offset];
    }
    return static_cast<const int64_t*>(k->data())[offset];
  }

  double GetP(std::optional<Tensor> p, Tensor::Size row) const {
    if (!p.has_value()) return 1.0;

    const auto offset = (p->size(0) == 1 ? 0 : row) * p->stride(0);
    switch (p->dtype()) {
      case DataType::kFloat16:
        return Cast<float>(static_cast<const Float16*>(p->data())[offset]);
      case DataType::kBFloat16:
        return Cast<float>(static_cast<const BFloat16*>(p->data())[offset]);
      case DataType::kFloat32:
        return static_cast<const float*>(p->data())[offset];
      case DataType::kFloat64:
        return static_cast<const double*>(p->data())[offset];
      default:
        assert(false && "`TopKTopPSampler` has unsupported `p` dtype.");
        return 1.0;
    }
  }

  mutable std::mt19937 rng_{std::random_device{}()};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_CPU_TOP_K_TOP_P_SAMPLER_H_
