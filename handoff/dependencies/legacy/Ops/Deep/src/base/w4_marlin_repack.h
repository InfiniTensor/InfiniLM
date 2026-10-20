#pragma once
#include "base/w4_marlin_common.h"

namespace infini::ops {
// scheme=0: AWQ GEMM nibble order; scheme=1: GPTQ K-packed order.
// perm is a verified loading-time bijection; empty means identity.
class W4MarlinRepack : public Operator<W4MarlinRepack> {
 public:
  W4MarlinRepack(const Tensor qweight, const Tensor perm, const int64_t scheme,
                 const int64_t size_k, const int64_t size_n, Tensor out)
      : scheme_{scheme}, k_{static_cast<Tensor::Size>(size_k)}, n_{static_cast<Tensor::Size>(size_n)} {
    w4::dims(size_k, size_n);
    w4::require(scheme == 0 || scheme == 1, "unknown W4 checkpoint packing");
    w4::shape(qweight, scheme == 0 ? Tensor::Shape{k_, n_/8} : Tensor::Shape{k_/8, n_}, DataType::kInt32);
    w4::shape(out, {k_/16, n_*2}, DataType::kInt32);
    w4::shape(perm, {perm.numel() ? k_ : 0}, DataType::kInt32);
    w4::require(scheme == 1 || perm.numel() == 0, "AWQ cannot take a K permutation");
    w4::device(qweight, perm); w4::device(qweight, out);
    w4::disjoint(out, qweight); w4::disjoint(out, perm);
  }
  virtual void operator()(const Tensor qweight, const Tensor perm, const int64_t scheme,
                          const int64_t size_k, const int64_t size_n, Tensor out) const = 0;
 protected:
  int64_t scheme_;
  Tensor::Size k_, n_;
};
} // namespace infini::ops
