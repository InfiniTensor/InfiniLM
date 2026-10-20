#pragma once
#include <array>
#include <stdexcept>

#include "operator.h"
namespace infini::ops {
class Mrope : public Operator<Mrope> {
 public:
  Mrope(const Tensor q, const Tensor k, const Tensor cos, const Tensor sin,
        const Tensor p, int head, int rot, int st, int sh, int sw,
        bool interleaved, Tensor qo, Tensor ko) {
    if (head <= 0 || rot < 0 || rot > head || rot % 2 || st < 0 || sh < 0 ||
        sw < 0 || int64_t(st) + sh + sw != rot / 2)
      throw std::invalid_argument("Mrope invalid sections");
    for (const auto& t : {k, cos, sin, p, qo, ko})
      if (t.device() != q.device())
        throw std::invalid_argument("Mrope devices differ");
    for (const auto& t : {q, k, qo, ko})
      if (t.ndim() != 2 || t.stride(1) != 1 || t.size(0) != q.size(0) ||
          t.dtype() != q.dtype() || t.size(1) % head)
        throw std::invalid_argument("Mrope invalid q/k metadata");
    if (q.shape() != qo.shape() || k.shape() != ko.shape() ||
        (q.dtype() != DataType::kFloat16 && q.dtype() != DataType::kBFloat16 &&
         q.dtype() != DataType::kFloat32 && q.dtype() != DataType::kFloat64) ||
        cos.ndim() != 2 || sin.ndim() != 2 || cos.shape() != sin.shape() ||
        cos.size(1) != rot / 2 || cos.stride(1) != 1 || sin.stride(1) != 1 ||
        cos.dtype() != q.dtype() || sin.dtype() != q.dtype() ||
        (p.dtype() != DataType::kInt32 && p.dtype() != DataType::kInt64) ||
        !((p.ndim() == 1 && p.size(0) == q.size(0)) ||
          (p.ndim() == 2 && p.size(0) == 3 && p.size(1) == q.size(0))))
      throw std::invalid_argument("Mrope invalid tables/positions/outputs");
  }
  virtual void operator()(const Tensor q, const Tensor k, const Tensor cos,
                          const Tensor sin, const Tensor p, int head, int rot,
                          int st, int sh, int sw, bool interleaved, Tensor qo,
                          Tensor ko) const = 0;
};
}  // namespace infini::ops
