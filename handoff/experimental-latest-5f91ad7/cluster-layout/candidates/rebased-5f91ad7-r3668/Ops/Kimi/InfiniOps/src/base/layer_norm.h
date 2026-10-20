#ifndef INFINI_OPS_BASE_LAYER_NORM_H_
#define INFINI_OPS_BASE_LAYER_NORM_H_

#include <functional>
#include <optional>

#include "operator.h"

namespace infini::ops {

// Inference LayerNorm over the last dimension. The legacy Core training
// intermediates are deliberately not part of this operator's outputs.
class LayerNorm : public Operator<LayerNorm> {
 public:
  static constexpr bool kAutomaticDefault = true;

  LayerNorm(const Tensor input, const Tensor weight,
            const std::optional<Tensor> bias, float eps, Tensor out)
      : input_meta_{input},
        weight_meta_{weight},
        bias_meta_{bias},
        out_meta_{out},
        device_{input.device()},
        eps_{eps} {}

  virtual void operator()(const Tensor input, const Tensor weight,
                          const std::optional<Tensor> bias, float eps,
                          Tensor out) const = 0;

 protected:
  const Tensor input_meta_, weight_meta_;
  const std::optional<Tensor> bias_meta_;
  const Tensor out_meta_;
  Device device_;
  float eps_;
};

}  // namespace infini::ops

#endif
