#ifndef INFINI_OPS_BASE_TOP_K_TOP_P_SAMPLE_INFINILM_H_
#define INFINI_OPS_BASE_TOP_K_TOP_P_SAMPLE_INFINILM_H_

#include <cassert>
#include <cstdint>
#include <optional>

#include "data_type.h"
#include "operator.h"
#include "tensor.h"

namespace infini::ops {

/// \deprecated Migrate to an open-source-aligned operator when available. This
/// interface will be removed in a future release.
class [[deprecated(
    "Migrate to an open-source-aligned operator when available.")]]
TopKTopPSampleInfinilm : public Operator<TopKTopPSampleInfinilm> {
 public:
  TopKTopPSampleInfinilm(const Tensor logits, std::optional<Tensor> k,
                         std::optional<Tensor> p, uint64_t seed,
                         uint64_t offset, Tensor out)
      : batch_size_{logits.size(0)},
        vocab_size_{logits.size(1)},
        dtype_{logits.dtype()} {
    (void)seed;
    (void)offset;
    assert(logits.ndim() == 2 &&
           "`TopKTopPSampleInfinilm` requires 2D `[batch_size, vocab_size]` "
           "logits");
    assert((dtype_ == DataType::kFloat16 || dtype_ == DataType::kBFloat16 ||
            dtype_ == DataType::kFloat32 || dtype_ == DataType::kFloat64) &&
           "`TopKTopPSampleInfinilm` requires floating-point logits");
    assert(out.ndim() == 1 &&
           "`TopKTopPSampleInfinilm` requires 1D `[batch_size]` output");
    assert(out.size(0) == batch_size_ &&
           "`TopKTopPSampleInfinilm` requires output batch size to match "
           "logits");
    assert(out.dtype() == DataType::kInt32 &&
           "`TopKTopPSampleInfinilm` requires int32 output");

    ValidateK(k);
    ValidateP(p);
  }

  virtual void operator()(const Tensor logits, std::optional<Tensor> k,
                          std::optional<Tensor> p, uint64_t seed,
                          uint64_t offset, Tensor out) const = 0;

 protected:
  void ValidateK(std::optional<Tensor> k) const {
    if (!k.has_value()) return;

    assert(k->ndim() == 1 &&
           "`TopKTopPSampleInfinilm` requires `k` to be 1D when provided");
    assert((k->size(0) == 1 || k->size(0) == batch_size_) &&
           "`TopKTopPSampleInfinilm` requires `k` shape [1] or [batch_size]");
    assert((k->dtype() == DataType::kInt32 || k->dtype() == DataType::kInt64) &&
           "`TopKTopPSampleInfinilm` requires int32 or int64 `k`");
  }

  void ValidateP(std::optional<Tensor> p) const {
    if (!p.has_value()) return;

    assert(p->ndim() == 1 &&
           "`TopKTopPSampleInfinilm` requires `p` to be 1D when provided");
    assert((p->size(0) == 1 || p->size(0) == batch_size_) &&
           "`TopKTopPSampleInfinilm` requires `p` shape [1] or [batch_size]");
    assert((p->dtype() == DataType::kFloat16 ||
            p->dtype() == DataType::kBFloat16 ||
            p->dtype() == DataType::kFloat32 ||
            p->dtype() == DataType::kFloat64) &&
           "`TopKTopPSampleInfinilm` requires floating-point `p`");
  }

  Tensor::Size batch_size_{0};

  Tensor::Size vocab_size_{0};

  DataType dtype_;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_TOP_K_TOP_P_SAMPLE_INFINILM_H_
