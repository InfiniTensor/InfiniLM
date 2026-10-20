#pragma once
#include "base/w4_marlin_common.h"

namespace infini::ops {
// Scale and AWQ zero layout transformation is an Ops loading primitive.
// For GPTQ symmetric input, qzeros/zeros_out are empty after LM validates
// z+1=8.
class W4MarlinParameters : public Operator<W4MarlinParameters> {
 public:
  W4MarlinParameters(const Tensor scales, const Tensor qzeros,
                     const int64_t scheme, const int64_t size_k,
                     const int64_t size_n, const int64_t group_size,
                     Tensor scales_out, Tensor zeros_out)
      : scheme_{scheme},
        group_{group_size},
        k_{static_cast<Tensor::Size>(size_k)},
        n_{static_cast<Tensor::Size>(size_n)} {
    w4::dims(size_k, size_n);
    w4::groups(k_, group_);
    w4::require(scheme == 0 || scheme == 1, "unknown W4 parameter scheme");
    w4::require(w4::fp16(scales) && scales.ndim() == 2 && scales.size(0) > 0,
                "W4 scales must be a nonempty F16/BF16 matrix");
    const auto g = scales.size(0);
    w4::require(w4::mul(g, n_) <=
                    static_cast<std::size_t>(std::numeric_limits<int>::max()),
                "W4 scale indexing exceeds int32 limits");
    w4::require(
        scheme != 0 || g == static_cast<std::size_t>(w4::groups(k_, group_)),
        "AWQ scales rows mismatch group_size");
    w4::shape(scales, {g, n_}, scales.dtype());
    w4::shape(scales_out, {g, n_}, scales.dtype());
    w4::shape(qzeros, scheme == 0 ? Tensor::Shape{g, n_ / 8} : Tensor::Shape{0},
              DataType::kInt32);
    w4::shape(zeros_out,
              scheme == 0 ? Tensor::Shape{g, n_ / 8} : Tensor::Shape{0},
              DataType::kInt32);
    for (const auto& t : {scales, qzeros, scales_out, zeros_out})
      w4::device(scales, t);
    for (const auto& t : {scales, qzeros}) {
      w4::disjoint(scales_out, t);
      w4::disjoint(zeros_out, t);
    }
    w4::disjoint(scales_out, zeros_out);
  }
  virtual void operator()(const Tensor scales, const Tensor qzeros,
                          const int64_t scheme, const int64_t size_k,
                          const int64_t size_n, const int64_t group_size,
                          Tensor scales_out, Tensor zeros_out) const = 0;

 protected:
  int64_t scheme_, group_;
  Tensor::Size k_, n_;
};
}  // namespace infini::ops
