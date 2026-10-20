#ifndef INFINI_OPS_BASE_MOE_WNA16_GEMM_H_
#define INFINI_OPS_BASE_MOE_WNA16_GEMM_H_

#include <cassert>
#include <cstdint>
#include <limits>
#include <optional>

#include "operator.h"

namespace infini::ops {

// Aligned with vLLM `_moe_C::moe_wna16_gemm` at commit
// ffc4f08c8ee130d4ea6347c1bf31ffd4f8af28ab.
class MoeWna16Gemm : public Operator<MoeWna16Gemm> {
 public:
  MoeWna16Gemm(const Tensor input, const Tensor b_qweight,
               const Tensor b_scales, std::optional<Tensor> b_qzeros,
               std::optional<Tensor> topk_weights,
               const Tensor sorted_token_ids, const Tensor expert_ids,
               const Tensor num_tokens_post_pad, const int64_t top_k,
               const int64_t block_size_m, const int64_t block_size_n,
               const int64_t block_size_k, const int64_t bit, Tensor output)
      : num_experts_{b_qweight.ndim() == 3 ? b_qweight.size(0) : 0},
        m_{input.ndim() == 2 ? input.size(0) : 0},
        n_{b_qweight.ndim() == 3 ? b_qweight.size(1) : 0},
        k_{input.ndim() == 2 ? input.size(1) : 0},
        num_groups_{b_scales.ndim() == 3 ? b_scales.size(2) : 0},
        group_size_{num_groups_ > 0 && k_ % num_groups_ == 0 ? k_ / num_groups_
                                                             : 0},
        sorted_token_ids_size_{sorted_token_ids.numel()},
        expert_ids_size_{expert_ids.numel()},
        top_k_{top_k},
        block_size_m_{block_size_m},
        block_size_n_{block_size_n},
        block_size_k_{block_size_k},
        bit_{bit},
        dtype_{input.dtype()},
        has_qzeros_{b_qzeros.has_value()},
        has_topk_weights_{topk_weights.has_value()},
        device_type_{input.device().type()},
        device_index_{input.device().index()} {
    Validate(input, b_qweight, b_scales, b_qzeros, topk_weights,
             sorted_token_ids, expert_ids, num_tokens_post_pad, output);
  }

  virtual void operator()(
      const Tensor input, const Tensor b_qweight, const Tensor b_scales,
      std::optional<Tensor> b_qzeros, std::optional<Tensor> topk_weights,
      const Tensor sorted_token_ids, const Tensor expert_ids,
      const Tensor num_tokens_post_pad, const int64_t top_k,
      const int64_t block_size_m, const int64_t block_size_n,
      const int64_t block_size_k, const int64_t bit, Tensor output) const = 0;

 protected:
  void ValidateCallMetadata(
      const Tensor input, const Tensor b_qweight, const Tensor b_scales,
      std::optional<Tensor> b_qzeros, std::optional<Tensor> topk_weights,
      const Tensor sorted_token_ids, const Tensor expert_ids,
      const Tensor num_tokens_post_pad, const int64_t top_k,
      const int64_t block_size_m, const int64_t block_size_n,
      const int64_t block_size_k, const int64_t bit,
      const Tensor output) const {
    assert(top_k == top_k_ && block_size_m == block_size_m_ &&
           block_size_n == block_size_n_ && block_size_k == block_size_k_ &&
           bit == bit_ &&
           "`MoeWna16Gemm` attributes changed after descriptor creation");

    const auto same_device = [&](const Tensor tensor) {
      return tensor.device().type() == device_type_ &&
             tensor.device().index() == device_index_;
    };
    const auto values_per_byte = static_cast<Tensor::Size>(8 / bit_);
    const auto matches =
        input.ndim() == 2 && input.size(0) == m_ && input.size(1) == k_ &&
        input.dtype() == dtype_ && input.IsContiguous() && same_device(input) &&
        b_qweight.ndim() == 3 && b_qweight.size(0) == num_experts_ &&
        b_qweight.size(1) == n_ && b_qweight.size(2) == k_ / values_per_byte &&
        b_qweight.dtype() == DataType::kUInt8 && b_qweight.IsContiguous() &&
        same_device(b_qweight) && b_scales.ndim() == 3 &&
        b_scales.size(0) == num_experts_ && b_scales.size(1) == n_ &&
        b_scales.size(2) == num_groups_ && b_scales.dtype() == dtype_ &&
        b_scales.IsContiguous() && same_device(b_scales) &&
        sorted_token_ids.ndim() == 1 &&
        sorted_token_ids.numel() == sorted_token_ids_size_ &&
        sorted_token_ids.dtype() == DataType::kInt32 &&
        sorted_token_ids.IsContiguous() && same_device(sorted_token_ids) &&
        expert_ids.ndim() == 1 && expert_ids.numel() == expert_ids_size_ &&
        expert_ids.dtype() == DataType::kInt32 && expert_ids.IsContiguous() &&
        same_device(expert_ids) && num_tokens_post_pad.ndim() == 1 &&
        num_tokens_post_pad.numel() == 1 &&
        num_tokens_post_pad.dtype() == DataType::kInt32 &&
        num_tokens_post_pad.IsContiguous() &&
        same_device(num_tokens_post_pad) && output.ndim() == 3 &&
        output.size(0) == m_ && output.size(1) == top_k_ &&
        output.size(2) == n_ && output.dtype() == dtype_ &&
        output.IsContiguous() && same_device(output) &&
        b_qzeros.has_value() == has_qzeros_ &&
        topk_weights.has_value() == has_topk_weights_;
    assert(matches && "`MoeWna16Gemm` call metadata must match descriptor");

    if (b_qzeros) {
      assert(b_qzeros->ndim() == 3 && b_qzeros->size(0) == num_experts_ &&
             b_qzeros->size(1) == n_ / values_per_byte &&
             b_qzeros->size(2) == num_groups_ &&
             b_qzeros->dtype() == DataType::kUInt8 &&
             b_qzeros->IsContiguous() && same_device(*b_qzeros) &&
             "`MoeWna16Gemm` zero-point metadata must match descriptor");
    }

    if (topk_weights) {
      assert(topk_weights->ndim() == 2 && topk_weights->size(0) == m_ &&
             topk_weights->size(1) == top_k_ &&
             topk_weights->dtype() == DataType::kFloat32 &&
             topk_weights->IsContiguous() && same_device(*topk_weights) &&
             "`MoeWna16Gemm` top-k weight metadata must match descriptor");
    }
  }

  Tensor::Size num_experts_{0};

  Tensor::Size m_{0};

  Tensor::Size n_{0};

  Tensor::Size k_{0};

  Tensor::Size num_groups_{0};

  Tensor::Size group_size_{0};

  Tensor::Size sorted_token_ids_size_{0};

  Tensor::Size expert_ids_size_{0};

  int64_t top_k_{0};

  int64_t block_size_m_{0};

  int64_t block_size_n_{0};

  int64_t block_size_k_{0};

  int64_t bit_{0};

  DataType dtype_;

  bool has_qzeros_{false};

  bool has_topk_weights_{false};

  Device::Type device_type_;

  int device_index_{0};

 private:
  void Validate(const Tensor input, const Tensor b_qweight,
                const Tensor b_scales, std::optional<Tensor> b_qzeros,
                std::optional<Tensor> topk_weights,
                const Tensor sorted_token_ids, const Tensor expert_ids,
                const Tensor num_tokens_post_pad, const Tensor output) const {
    assert((dtype_ == DataType::kFloat16 || dtype_ == DataType::kBFloat16) &&
           "`MoeWna16Gemm` supports float16 and bfloat16 inputs");
    assert((bit_ == 4 || bit_ == 8) &&
           "`MoeWna16Gemm` requires 4-bit or 8-bit weights");
    assert(m_ > 0 && n_ > 0 && k_ > 0 && num_experts_ > 0 && top_k_ > 0 &&
           "`MoeWna16Gemm` requires positive dimensions");
    assert(num_groups_ > 0 && group_size_ > 0 &&
           "`MoeWna16Gemm` requires an integral quantization group size");

    const auto values_per_byte = static_cast<Tensor::Size>(8 / bit_);
    const auto values_per_word = static_cast<Tensor::Size>(32 / bit_);
    assert(k_ % values_per_byte == 0 && n_ % values_per_word == 0 &&
           group_size_ % values_per_word == 0 &&
           "`MoeWna16Gemm` packed dimensions are incompatible with `bit`");
    assert(block_size_m_ > 0 && block_size_m_ <= 64 && block_size_n_ > 0 &&
           block_size_n_ <= 1024 && block_size_k_ > 0 && block_size_k_ <= k_ &&
           block_size_k_ % group_size_ == 0 && k_ % block_size_k_ == 0 &&
           block_size_n_ % values_per_word == 0 &&
           (block_size_k_ / values_per_word) % 4 == 0 &&
           "`MoeWna16Gemm` received unsupported block sizes");
    const auto groups_per_block = block_size_k_ / group_size_;
    assert((groups_per_block == 1 || groups_per_block == 2 ||
            groups_per_block == 4 || groups_per_block == 8) &&
           "`MoeWna16Gemm` requires 1, 2, 4, or 8 groups per K block");

    constexpr auto kMaxU16 =
        static_cast<Tensor::Size>(std::numeric_limits<uint16_t>::max());
    constexpr auto kMaxU32 =
        static_cast<Tensor::Size>(std::numeric_limits<uint32_t>::max());
    assert(num_experts_ <= kMaxU16 && group_size_ <= kMaxU16 &&
           top_k_ <= static_cast<int64_t>(kMaxU16) &&
           block_size_m_ <= static_cast<int64_t>(kMaxU16) &&
           block_size_n_ <= static_cast<int64_t>(kMaxU16) &&
           block_size_k_ <= static_cast<int64_t>(kMaxU16) && m_ <= kMaxU32 &&
           n_ <= kMaxU32 && k_ <= kMaxU32 &&
           "`MoeWna16Gemm` dimensions exceed CUDA kernel limits");
    assert(m_ <= std::numeric_limits<Tensor::Size>::max() / top_k_ &&
           m_ * top_k_ <= std::numeric_limits<Tensor::Size>::max() / n_ &&
           "`MoeWna16Gemm` output dimensions overflow");

    auto effective_sorted_size = sorted_token_ids_size_;
    if (m_ <= block_size_m_) {
      const auto limit = m_ * block_size_m_ * top_k_;
      if (effective_sorted_size > limit) {
        effective_sorted_size = limit;
      }
    }
    const auto num_token_blocks =
        (effective_sorted_size + block_size_m_ - 1) / block_size_m_;
    assert(sorted_token_ids_size_ > 0 && expert_ids_size_ >= num_token_blocks &&
           "`MoeWna16Gemm` routing metadata is too small");
    assert(num_token_blocks <= kMaxU32 &&
           "`MoeWna16Gemm` token block count exceeds CUDA grid limits");
    assert((n_ + block_size_n_ - 1) / block_size_n_ <= 65535 &&
           (k_ + block_size_k_ - 1) / block_size_k_ <= 65535 &&
           "`MoeWna16Gemm` grid dimensions exceed CUDA limits");

    ValidateCallMetadata(input, b_qweight, b_scales, b_qzeros, topk_weights,
                         sorted_token_ids, expert_ids, num_tokens_post_pad,
                         top_k_, block_size_m_, block_size_n_, block_size_k_,
                         bit_, output);
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_MOE_WNA16_GEMM_H_
