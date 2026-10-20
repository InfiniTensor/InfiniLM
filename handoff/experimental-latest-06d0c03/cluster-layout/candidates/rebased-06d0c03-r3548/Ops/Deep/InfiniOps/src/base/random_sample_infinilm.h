#ifndef INFINI_OPS_BASE_RANDOM_SAMPLE_INFINILM_H_
#define INFINI_OPS_BASE_RANDOM_SAMPLE_INFINILM_H_

#include <cassert>
#include <cstdint>
#include <limits>

#include "data_type.h"
#include "operator.h"
#include "tensor.h"

namespace infini::ops {

/// \deprecated Migrate to an open-source-aligned operator when available. This
/// interface will be removed in a future release.
class [[deprecated(
    "Migrate to an open-source-aligned operator when available.")]]
RandomSampleInfinilm : public Operator<RandomSampleInfinilm> {
 public:
  RandomSampleInfinilm(const Tensor logits, float random_val, float topp,
                       int64_t topk, float temperature, Tensor out)
      : dtype_{logits.dtype()},
        out_dtype_{out.dtype()},
        n_{logits.size(0)},
        logits_stride_{logits.stride(0)},
        topp_{topp},
        topk_{topk},
        temperature_{temperature} {
    assert(logits.ndim() == 1 && "`RandomSampleInfinilm` requires 1D logits");
    assert(n_ > 0 && "`RandomSampleInfinilm` requires non-empty logits");
    assert(logits.stride(0) == 1 &&
           "`RandomSampleInfinilm` requires contiguous logits");
    assert(out.numel() == 1 && "`RandomSampleInfinilm` requires scalar output");
    assert(IsFloatDtype(dtype_) &&
           "`RandomSampleInfinilm` requires floating-point logits");
    assert(IsIntDtype(out_dtype_) &&
           "`RandomSampleInfinilm` requires integer output");
    assert(topk > 0 && "`RandomSampleInfinilm` requires `topk > 0`");
    assert(topk <= std::numeric_limits<int>::max() &&
           "`RandomSampleInfinilm` requires `topk` to fit in int");
  }

  virtual void operator()(const Tensor logits, float random_val, float topp,
                          int64_t topk, float temperature,
                          Tensor out) const = 0;

 protected:
  static bool IsFloatDtype(DataType dtype) {
    return dtype == DataType::kFloat16 || dtype == DataType::kBFloat16 ||
           dtype == DataType::kFloat32 || dtype == DataType::kFloat64;
  }

  static bool IsIntDtype(DataType dtype) {
    return dtype == DataType::kInt8 || dtype == DataType::kInt16 ||
           dtype == DataType::kInt32 || dtype == DataType::kInt64 ||
           dtype == DataType::kUInt8 || dtype == DataType::kUInt16 ||
           dtype == DataType::kUInt32 || dtype == DataType::kUInt64;
  }

  DataType dtype_;

  DataType out_dtype_;

  Tensor::Size n_{0};

  Tensor::Stride logits_stride_{1};

  float topp_{0.0f};

  int64_t topk_{1};

  float temperature_{1.0f};
};

template <>
struct CacheKeyBuilder<RandomSampleInfinilm> {
  detail::CacheKey operator()(const Config& config, const Tensor logits,
                              float /*random_val*/, float topp, int64_t topk,
                              float temperature, Tensor out) const {
    return detail::CacheKey::Build(config.implementation_index(), logits, topp,
                                   topk, temperature, out);
  }
};

}  // namespace infini::ops

#endif
