#pragma once
#include "base/w4_marlin_gemm.h"
#include "base/w4_marlin_parameters.h"
#include "base/w4_marlin_repack.h"

namespace infini::ops {
template <>
class Operator<W4MarlinRepack, Device::Type::kNvidia, 0>
    : public W4MarlinRepack {
 public:
  Operator(const Tensor qweight, const Tensor perm, const int64_t scheme,
           const int64_t size_k, const int64_t size_n, Tensor out);
  using W4MarlinRepack::operator();
  void operator()(const Tensor qweight, const Tensor perm, const int64_t scheme,
                  const int64_t size_k, const int64_t size_n,
                  Tensor out) const override;

 private:
  int sms_{0}, shared_{0}, device_{0};
};
template <>
class Operator<W4MarlinParameters, Device::Type::kNvidia, 0>
    : public W4MarlinParameters {
 public:
  Operator(const Tensor scales, const Tensor qzeros, const int64_t scheme,
           const int64_t size_k, const int64_t size_n, const int64_t group_size,
           Tensor scales_out, Tensor zeros_out);
  using W4MarlinParameters::operator();
  void operator()(const Tensor scales, const Tensor qzeros,
                  const int64_t scheme, const int64_t size_k,
                  const int64_t size_n, const int64_t group_size,
                  Tensor scales_out, Tensor zeros_out) const override;

 private:
  int device_{0};
};
template <>
class Operator<W4MarlinGemm, Device::Type::kNvidia, 0> : public W4MarlinGemm {
 public:
  Operator(const Tensor a, const Tensor b, const Tensor scales,
           const Tensor zeros, const Tensor g_idx, const Tensor perm,
           const Tensor bias, const int64_t scheme, const int64_t group_size,
           const bool is_k_full, const float alpha, Tensor out);
  using W4MarlinGemm::operator();
  void operator()(const Tensor a, const Tensor b, const Tensor scales,
                  const Tensor zeros, const Tensor g_idx, const Tensor perm,
                  const Tensor bias, const int64_t scheme,
                  const int64_t group_size, const bool is_k_full,
                  const float alpha, Tensor out) const override;
  std::size_t workspace_size_in_bytes() const override { return bytes_; }
  // Original host selector still sets CUDA function attributes and queries the
  // device at invocation. No capture promise without a qualified frozen launch.

 private:
  int device_{0}, sms_{0};
  std::size_t ctmp_{0}, atmp_{0}, locks_{0}, bytes_{0};
};
}  // namespace infini::ops
