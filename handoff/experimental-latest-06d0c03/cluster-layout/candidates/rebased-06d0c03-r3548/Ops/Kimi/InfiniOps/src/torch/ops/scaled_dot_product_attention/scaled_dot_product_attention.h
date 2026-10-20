#ifndef INFINI_OPS_TORCH_SCALED_DOT_PRODUCT_ATTENTION_H_
#define INFINI_OPS_TORCH_SCALED_DOT_PRODUCT_ATTENTION_H_

#include <optional>

#include "base/scaled_dot_product_attention.h"

namespace infini::ops {

template <Device::Type kDev>
class Operator<ScaledDotProductAttention, kDev, 1>
    : public ScaledDotProductAttention {
 public:
  Operator(const Tensor query, const Tensor key, const Tensor value,
           const std::optional<Tensor> attn_mask, double dropout_p,
           bool is_causal, const std::optional<double> scale, bool enable_gqa,
           Tensor out);

  Operator(const Tensor query, const Tensor key, const Tensor value,
           Tensor out);

  void operator()(const Tensor query, const Tensor key, const Tensor value,
                  const std::optional<Tensor> attn_mask, double dropout_p,
                  bool is_causal, const std::optional<double> scale,
                  bool enable_gqa, Tensor out) const override;
};

}  // namespace infini::ops

#endif
