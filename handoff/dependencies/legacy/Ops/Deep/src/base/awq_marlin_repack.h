#ifndef INFINI_OPS_BASE_AWQ_MARLIN_REPACK_H_
#define INFINI_OPS_BASE_AWQ_MARLIN_REPACK_H_

#include <cassert>
#include <cstdint>
#include <functional>
#include <limits>

#include "operator.h"

namespace infini::ops {

// Aligned with vLLM's low-level `awq_marlin_repack` operator.
class AwqMarlinRepack : public Operator<AwqMarlinRepack> {
 public:
  AwqMarlinRepack(const Tensor b_q_weight, const int64_t size_k,
                  const int64_t size_n, const int64_t num_bits,
                  const bool is_a_8bit, Tensor out)
      : b_q_weight_metadata_{b_q_weight},
        out_metadata_{out},
        size_k_{size_k},
        size_n_{size_n},
        num_bits_{num_bits},
        pack_factor_{num_bits == 4   ? 8
                     : num_bits == 8 ? 4
                                     : 0},
        is_a_8bit_{is_a_8bit},
        device_index_{b_q_weight.device().index()} {
    Validate(b_q_weight, out);
  }

  virtual void operator()(const Tensor b_q_weight, const int64_t size_k,
                          const int64_t size_n, const int64_t num_bits,
                          const bool is_a_8bit, Tensor out) const = 0;

 protected:
  void ValidateCallMetadata(const Tensor b_q_weight, const int64_t size_k,
                            const int64_t size_n, const int64_t num_bits,
                            const bool is_a_8bit, const Tensor out) const {
    assert(size_k == size_k_ && size_n == size_n_ && num_bits == num_bits_ &&
           is_a_8bit == is_a_8bit_ &&
           "`AwqMarlinRepack` attributes changed after descriptor creation");

    const std::equal_to<Tensor> same_metadata;
    assert(same_metadata(b_q_weight_metadata_, b_q_weight) &&
           same_metadata(out_metadata_, out) &&
           "`AwqMarlinRepack` tensor metadata differs from its descriptor");
  }

  Tensor b_q_weight_metadata_;

  Tensor out_metadata_;

  int64_t size_k_{0};

  int64_t size_n_{0};

  int64_t num_bits_{0};

  int64_t pack_factor_{0};

  bool is_a_8bit_{false};

  int device_index_{0};

 private:
  void Validate(const Tensor b_q_weight, const Tensor out) const {
    assert((num_bits_ == 4 || num_bits_ == 8) &&
           "`AwqMarlinRepack` requires `num_bits` to be 4 or 8");
    assert(size_k_ > 0 && size_n_ > 0 && size_k_ % 16 == 0 &&
           size_n_ % 64 == 0 &&
           "`AwqMarlinRepack` requires positive `size_k` divisible by 16 "
           "and `size_n` divisible by 64");
    assert((!is_a_8bit_ || size_k_ % 32 == 0) &&
           "`AwqMarlinRepack` requires `size_k` divisible by 32 for A8 "
           "layouts");
    assert(size_k_ <= std::numeric_limits<int>::max() &&
           size_n_ <= std::numeric_limits<int>::max() &&
           size_k_ <=
               std::numeric_limits<int>::max() / (size_n_ / pack_factor_) &&
           "`AwqMarlinRepack` dimensions exceed CUDA kernel limits");

    assert(b_q_weight.ndim() == 2 &&
           b_q_weight.size(0) == static_cast<Tensor::Size>(size_k_) &&
           b_q_weight.size(1) ==
               static_cast<Tensor::Size>(size_n_ / pack_factor_) &&
           "`AwqMarlinRepack` requires `b_q_weight` shape "
           "[`size_k`, `size_n / pack_factor`]");
    assert(b_q_weight.dtype() == DataType::kInt32 &&
           b_q_weight.IsContiguous() &&
           "`AwqMarlinRepack` requires contiguous int32 `b_q_weight`");

    const Tensor::Shape expected_out_shape{
        static_cast<Tensor::Size>(size_k_ / 16),
        static_cast<Tensor::Size>(size_n_ * 16 / pack_factor_)};
    assert(out.shape() == expected_out_shape &&
           "`AwqMarlinRepack` output shape is incorrect");
    assert(out.dtype() == DataType::kInt32 && out.IsContiguous() &&
           "`AwqMarlinRepack` requires contiguous int32 output");
    assert(out.device().type() == b_q_weight.device().type() &&
           out.device().index() == b_q_weight.device().index() &&
           "`AwqMarlinRepack` requires input and output on the same device");
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_AWQ_MARLIN_REPACK_H_
