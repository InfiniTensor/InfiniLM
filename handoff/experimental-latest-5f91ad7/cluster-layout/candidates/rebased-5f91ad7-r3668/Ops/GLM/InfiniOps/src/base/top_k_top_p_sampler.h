#ifndef INFINI_OPS_BASE_TOP_K_TOP_P_SAMPLER_H_
#define INFINI_OPS_BASE_TOP_K_TOP_P_SAMPLER_H_

#include <cassert>
#include <optional>

#include "data_type.h"
#include "operator.h"
#include "tensor.h"

namespace infini::ops {

// Legacy sampler for 2D `logits` after optional rank and nucleus filtering.
// Temperature scaling is intentionally handled by callers.
// The optional `k` and `p` tensors may be shaped as `[1]` or `[batch_size]`.
/// \deprecated Migrate to an open-source-aligned operator when available. This
/// interface will be removed in a future release.
class [[deprecated(
    "Migrate to an open-source-aligned operator when available.")]]
TopKTopPSampler : public Operator<TopKTopPSampler> {
 public:
  TopKTopPSampler(const Tensor logits, std::optional<Tensor> k,
                  std::optional<Tensor> p, Tensor out)
      : batch_size_{logits.size(0)},
        vocab_size_{logits.size(1)},
        dtype_{logits.dtype()} {
    assert(logits.ndim() == 2 &&
           "`TopKTopPSampler` requires 2D `[batch_size, vocab_size]` logits");
    assert((dtype_ == DataType::kFloat16 || dtype_ == DataType::kBFloat16 ||
            dtype_ == DataType::kFloat32 || dtype_ == DataType::kFloat64) &&
           "`TopKTopPSampler` requires floating-point logits");
    assert(out.ndim() == 1 &&
           "`TopKTopPSampler` requires 1D `[batch_size]` output");
    assert(out.size(0) == batch_size_ &&
           "`TopKTopPSampler` requires output batch size to match logits");
    assert(out.dtype() == DataType::kInt32 &&
           "`TopKTopPSampler` requires int32 output");

    ValidateK(k);
    ValidateP(p);
  }

  virtual void operator()(const Tensor logits, std::optional<Tensor> k,
                          std::optional<Tensor> p, Tensor out) const = 0;

 protected:
  void ValidateK(std::optional<Tensor> k) const {
    if (!k.has_value()) return;

    assert(k->ndim() == 1 &&
           "`TopKTopPSampler` requires `k` to be 1D when provided");
    assert((k->size(0) == 1 || k->size(0) == batch_size_) &&
           "`TopKTopPSampler` requires `k` shape [1] or [batch_size]");
    assert((k->dtype() == DataType::kInt32 || k->dtype() == DataType::kInt64) &&
           "`TopKTopPSampler` requires int32 or int64 `k`");
  }

  void ValidateP(std::optional<Tensor> p) const {
    if (!p.has_value()) return;

    assert(p->ndim() == 1 &&
           "`TopKTopPSampler` requires `p` to be 1D when provided");
    assert((p->size(0) == 1 || p->size(0) == batch_size_) &&
           "`TopKTopPSampler` requires `p` shape [1] or [batch_size]");
    assert((p->dtype() == DataType::kFloat16 ||
            p->dtype() == DataType::kBFloat16 ||
            p->dtype() == DataType::kFloat32 ||
            p->dtype() == DataType::kFloat64) &&
           "`TopKTopPSampler` requires floating-point `p`");
  }

  Tensor::Size batch_size_{0};

  Tensor::Size vocab_size_{0};

  DataType dtype_;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_TOP_K_TOP_P_SAMPLER_H_
