#ifndef INFINI_OPS_NVIDIA_CUTLASS_SCALED_MM_CUTLASS_H_
#define INFINI_OPS_NVIDIA_CUTLASS_SCALED_MM_CUTLASS_H_

#include <optional>

#include "base/cutlass_scaled_mm.h"

namespace infini::ops {

template <>
class Operator<CutlassScaledMm, Device::Type::kNvidia, 0>
    : public CutlassScaledMm {
 public:
  Operator(const Tensor a, const Tensor b, const Tensor scale_a,
           const Tensor scale_b, std::optional<Tensor> bias,
           const DataType out_dtype, Tensor out);

  void operator()(const Tensor a, const Tensor b, const Tensor scale_a,
                  const Tensor scale_b, std::optional<Tensor> bias,
                  const DataType out_dtype, Tensor out) const override;

 private:
  int device_index_{0};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_NVIDIA_CUTLASS_SCALED_MM_CUTLASS_H_
