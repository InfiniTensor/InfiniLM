#ifndef INFINI_OPS_BASE_GPTQ_MARLIN_REPACK_H_
#define INFINI_OPS_BASE_GPTQ_MARLIN_REPACK_H_

#include <cassert>
#include <cstdint>
#include <functional>
#include <limits>

#include "operator.h"

namespace infini::ops {

// Aligned with vLLM's low-level `gptq_marlin_repack` operator.
class GptqMarlinRepack : public Operator<GptqMarlinRepack> {
 public:
  GptqMarlinRepack(const Tensor b_q_weight, const Tensor perm,
                   const int64_t size_k, const int64_t size_n,
                   const int64_t num_bits, const bool is_a_8bit, Tensor out)
      : b_q_weight_metadata_{b_q_weight},
        perm_metadata_{perm},
        out_metadata_{out},
        size_k_{size_k},
        size_n_{size_n},
        num_bits_{num_bits},
        pack_factor_{num_bits == 4   ? 8
                     : num_bits == 8 ? 4
                                     : 0},
        is_a_8bit_{is_a_8bit},
        has_perm_{perm.numel() != 0},
        device_index_{b_q_weight.device().index()} {
    Validate(b_q_weight, perm, out);
  }

  virtual void operator()(const Tensor b_q_weight, const Tensor perm,
                          const int64_t size_k, const int64_t size_n,
                          const int64_t num_bits, const bool is_a_8bit,
                          Tensor out) const = 0;

 protected:
  void ValidateCallMetadata(const Tensor b_q_weight, const Tensor perm,
                            const int64_t size_k, const int64_t size_n,
                            const int64_t num_bits, const bool is_a_8bit,
                            const Tensor out) const {
    assert(size_k == size_k_ && size_n == size_n_ && num_bits == num_bits_ &&
           is_a_8bit == is_a_8bit_ &&
           "`GptqMarlinRepack` attributes changed after descriptor creation");

    const std::equal_to<Tensor> same_metadata;
    assert(same_metadata(b_q_weight_metadata_, b_q_weight) &&
           same_metadata(perm_metadata_, perm) &&
           same_metadata(out_metadata_, out) &&
           "`GptqMarlinRepack` tensor metadata differs from its descriptor");
  }

  Tensor b_q_weight_metadata_;

  Tensor perm_metadata_;

  Tensor out_metadata_;

  int64_t size_k_{0};

  int64_t size_n_{0};

  int64_t num_bits_{0};

  int64_t pack_factor_{0};

  bool is_a_8bit_{false};

  bool has_perm_{false};

  int device_index_{0};

 private:
  void Validate(const Tensor b_q_weight, const Tensor perm,
                const Tensor out) const {
    assert((num_bits_ == 4 || num_bits_ == 8) &&
           "`GptqMarlinRepack` requires `num_bits` to be 4 or 8");
    assert(size_k_ > 0 && size_n_ > 0 && size_k_ % 16 == 0 &&
           size_n_ % 64 == 0 &&
           "`GptqMarlinRepack` requires positive `size_k` divisible by 16 "
           "and `size_n` divisible by 64");
    assert((!is_a_8bit_ || size_k_ % 32 == 0) &&
           "`GptqMarlinRepack` requires `size_k` divisible by 32 for A8 "
           "layouts");
    assert(size_k_ <= std::numeric_limits<int>::max() &&
           size_n_ <= std::numeric_limits<int>::max() &&
           size_n_ <= std::numeric_limits<int64_t>::max() / 16 &&
           "`GptqMarlinRepack` dimensions exceed CUDA kernel limits");

    assert(b_q_weight.ndim() == 2 &&
           b_q_weight.size(0) ==
               static_cast<Tensor::Size>(size_k_ / pack_factor_) &&
           b_q_weight.size(1) == static_cast<Tensor::Size>(size_n_) &&
           "`GptqMarlinRepack` requires `b_q_weight` shape "
           "[`size_k / pack_factor`, `size_n`]");
    assert(b_q_weight.dtype() == DataType::kInt32 &&
           b_q_weight.IsContiguous() &&
           "`GptqMarlinRepack` requires contiguous int32 `b_q_weight`");

    assert(perm.ndim() == 1 &&
           (!has_perm_ || perm.numel() == static_cast<Tensor::Size>(size_k_)) &&
           "`GptqMarlinRepack` requires empty `perm` or shape [`size_k`]");
    assert(perm.dtype() == DataType::kInt32 && perm.IsContiguous() &&
           "`GptqMarlinRepack` requires contiguous int32 `perm`");
    assert((!is_a_8bit_ || !has_perm_) &&
           "`GptqMarlinRepack` does not support `perm` for A8 layouts");

    const Tensor::Shape expected_out_shape{
        static_cast<Tensor::Size>(size_k_ / 16),
        static_cast<Tensor::Size>(size_n_ * 16 / pack_factor_)};
    assert(out.shape() == expected_out_shape &&
           "`GptqMarlinRepack` output shape is incorrect");
    assert(out.dtype() == DataType::kInt32 && out.IsContiguous() &&
           "`GptqMarlinRepack` requires contiguous int32 output");

    const auto same_device = [&](const Tensor tensor) {
      return tensor.device().type() == b_q_weight.device().type() &&
             tensor.device().index() == b_q_weight.device().index();
    };
    assert(same_device(perm) && same_device(out) &&
           "`GptqMarlinRepack` requires all tensors on the same device");
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_GPTQ_MARLIN_REPACK_H_
