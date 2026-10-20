#pragma once
#include "base/w4_marlin_common.h"

namespace infini::ops {
// Already repacked B. Y = alpha * Marlin(A,B,S,Z) + ordinary-order bias.
// No dense weight allocation; one stream, one external workspace per
// invocation.
class W4MarlinGemm : public Operator<W4MarlinGemm> {
 public:
  W4MarlinGemm(const Tensor a, const Tensor b, const Tensor scales,
               const Tensor zeros, const Tensor g_idx, const Tensor perm,
               const Tensor bias, const int64_t scheme,
               const int64_t group_size, const bool is_k_full,
               const float alpha, Tensor out)
      : scheme_{scheme}, group_{group_size}, full_{is_k_full}, alpha_{alpha} {
    w4::require(a.ndim() == 2 && out.ndim() == 2, "W4 GEMM requires matrices");
    k_ = a.size(1);
    n_ = out.size(1);
    m_ = a.size(0);
    w4::dims(k_, n_);
    const auto groups = w4::groups(k_, group_);
    w4::require(m_ > 0 && m_ <= std::numeric_limits<int>::max() && w4::fp16(a),
                "W4 activation must be nonempty F16/BF16");
    w4::require(a.stride(1) == 1 && a.stride(0) >= k_ && a.stride(0) % 8 == 0 &&
                    a.stride(0) <= std::numeric_limits<int>::max(),
                "W4 A requires aligned contiguous rows");
    w4::require(scheme == 0 || scheme == 1, "unknown W4 GEMM scheme");
    w4::require(std::isfinite(alpha), "W4 alpha must be finite");
    w4::shape(out, {m_, n_}, a.dtype());
    w4::shape(b, {k_ / 16, n_ * 2}, DataType::kInt32);
    w4::require(scales.ndim() == 2 && scales.size(0) > 0 &&
                    scales.size(0) <= std::numeric_limits<int>::max(),
                "W4 scales rows are invalid");
    const auto g = scales.size(0);
    w4::shape(scales, {g, n_}, a.dtype());
    w4::shape(zeros, scheme == 0 ? Tensor::Shape{g, n_ / 8} : Tensor::Shape{0},
              DataType::kInt32);
    w4::shape(g_idx, {g_idx.numel() ? k_ : 0}, DataType::kInt32);
    w4::shape(perm, {perm.numel() ? k_ : 0}, DataType::kInt32);
    w4::shape(bias, {bias.numel() ? n_ : 0}, a.dtype());
    const bool act = perm.numel() != 0;
    w4::require(act == (g_idx.numel() != 0) && (scheme == 1 || !act),
                "W4 g_idx/perm pairing is invalid");
    w4::require(full_ || act, "partial-K W4 requires GPTQ activation order");
    w4::require(!full_ || g == groups,
                "W4 full-K group count differs from checkpoint");
    w4::require(!act || group_ != -1,
                "activation order requires grouped quantization");
    w4::require(!act || !full_ || g > 1,
                "full-K activation order requires multiple groups");
    w4::require(full_ || static_cast<std::size_t>(group_) < k_,
                "partial-K scale ordering requires group_size < K");
    w4::require(
        w4::mul(m_, k_) <=
                static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
            w4::mul(m_, n_) <=
                static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
            w4::mul(g, n_) <=
                static_cast<std::size_t>(std::numeric_limits<int>::max()),
        "W4 indexing exceeds original int32 kernel limits");
    for (const auto& t : {a, b, scales, zeros, out})
      if (t.data()) w4::address(t, 16);
    for (const auto& t : {g_idx, perm, bias})
      if (t.data()) w4::address(t, 4);
    for (const auto& t : {a, b, scales, zeros, g_idx, perm, bias, out})
      w4::device(a, t);
    for (const auto& t : {a, b, scales, zeros, g_idx, perm, bias})
      w4::disjoint(out, t);
  }
  virtual void operator()(const Tensor a, const Tensor b, const Tensor scales,
                          const Tensor zeros, const Tensor g_idx,
                          const Tensor perm, const Tensor bias,
                          const int64_t scheme, const int64_t group_size,
                          const bool is_k_full, const float alpha,
                          Tensor out) const = 0;

 protected:
  int64_t scheme_, group_;
  Tensor::Size k_, n_, m_;
  bool full_;
  float alpha_;
};
}  // namespace infini::ops
