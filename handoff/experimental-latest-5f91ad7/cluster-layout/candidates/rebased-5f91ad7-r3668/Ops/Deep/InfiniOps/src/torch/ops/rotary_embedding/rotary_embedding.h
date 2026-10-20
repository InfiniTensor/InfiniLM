#ifndef INFINI_OPS_TORCH_ROTARY_EMBEDDING_H_
#define INFINI_OPS_TORCH_ROTARY_EMBEDDING_H_

#include <optional>

#include "base/rotary_embedding.h"

namespace infini::ops {

template <Device::Type kDev>
class Operator<RotaryEmbedding, kDev, 1> : public RotaryEmbedding {
 public:
  Operator(const Tensor positions, Tensor query, std::optional<Tensor> key,
           const Tensor cos_sin_cache, int64_t head_size, bool is_neox,
           int64_t rope_dim_offset = 0, bool inverse = false);

  void operator()(const Tensor positions, Tensor query,
                  std::optional<Tensor> key, const Tensor cos_sin_cache,
                  int64_t head_size, bool is_neox, int64_t rope_dim_offset = 0,
                  bool inverse = false) const override;
};

}  // namespace infini::ops

#endif
