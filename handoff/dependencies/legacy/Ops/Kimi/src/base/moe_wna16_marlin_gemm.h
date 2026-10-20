#ifndef INFINI_OPS_BASE_MOE_WNA16_MARLIN_GEMM_H_
#define INFINI_OPS_BASE_MOE_WNA16_MARLIN_GEMM_H_

#include <cassert>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>

#include "operator.h"

namespace infini::ops {

// Aligned with vLLM `_moe_C::moe_wna16_marlin_gemm` at commit
// 568afb3a13806beb53bb2e6bd518269357b237c0.
class MoeWna16MarlinGemm : public Operator<MoeWna16MarlinGemm> {
 public:
  MoeWna16MarlinGemm(const Tensor a, const Tensor b_q_weight,
                     std::optional<Tensor> b_bias_or_none,
                     const Tensor b_scales, std::optional<Tensor> a_scales,
                     std::optional<Tensor> global_scale,
                     std::optional<Tensor> b_zeros_or_none,
                     std::optional<Tensor> g_idx_or_none,
                     std::optional<Tensor> perm_or_none, const Tensor workspace,
                     const Tensor sorted_token_ids, const Tensor expert_ids,
                     const Tensor num_tokens_past_padded,
                     const Tensor topk_weights, const int64_t moe_block_size,
                     const int64_t top_k, const bool mul_topk_weights,
                     const int64_t b_type_id, const int64_t size_m,
                     const int64_t size_n, const int64_t size_k,
                     const bool is_full_k, const bool use_atomic_add,
                     const bool use_fp32_reduce, const bool is_zp_float,
                     const int64_t thread_k, const int64_t thread_n,
                     const int64_t blocks_per_sm, Tensor out)
      : a_metadata_{a},
        b_q_weight_metadata_{b_q_weight},
        b_bias_or_none_metadata_{b_bias_or_none},
        b_scales_metadata_{b_scales},
        a_scales_metadata_{a_scales},
        global_scale_metadata_{global_scale},
        b_zeros_or_none_metadata_{b_zeros_or_none},
        g_idx_or_none_metadata_{g_idx_or_none},
        perm_or_none_metadata_{perm_or_none},
        workspace_metadata_{workspace},
        sorted_token_ids_metadata_{sorted_token_ids},
        expert_ids_metadata_{expert_ids},
        num_tokens_past_padded_metadata_{num_tokens_past_padded},
        topk_weights_metadata_{topk_weights},
        out_metadata_{out},
        moe_block_size_{moe_block_size},
        top_k_{top_k},
        mul_topk_weights_{mul_topk_weights},
        b_type_id_{b_type_id},
        size_m_{size_m},
        size_n_{size_n},
        size_k_{size_k},
        is_full_k_{is_full_k},
        use_atomic_add_{use_atomic_add},
        use_fp32_reduce_{use_fp32_reduce},
        is_zp_float_{is_zp_float},
        thread_k_{thread_k},
        thread_n_{thread_n},
        blocks_per_sm_{blocks_per_sm},
        device_index_{a.device().index()} {
    Validate(a, b_q_weight, b_bias_or_none, b_scales, a_scales, global_scale,
             b_zeros_or_none, g_idx_or_none, perm_or_none, workspace,
             sorted_token_ids, expert_ids, num_tokens_past_padded, topk_weights,
             out);
  }

  virtual void operator()(
      const Tensor a, const Tensor b_q_weight,
      std::optional<Tensor> b_bias_or_none, const Tensor b_scales,
      std::optional<Tensor> a_scales, std::optional<Tensor> global_scale,
      std::optional<Tensor> b_zeros_or_none,
      std::optional<Tensor> g_idx_or_none, std::optional<Tensor> perm_or_none,
      const Tensor workspace, const Tensor sorted_token_ids,
      const Tensor expert_ids, const Tensor num_tokens_past_padded,
      const Tensor topk_weights, const int64_t moe_block_size,
      const int64_t top_k, const bool mul_topk_weights, const int64_t b_type_id,
      const int64_t size_m, const int64_t size_n, const int64_t size_k,
      const bool is_full_k, const bool use_atomic_add,
      const bool use_fp32_reduce, const bool is_zp_float,
      const int64_t thread_k, const int64_t thread_n,
      const int64_t blocks_per_sm, Tensor out) const = 0;

 protected:
  void ValidateCallMetadata(
      const Tensor a, const Tensor b_q_weight,
      std::optional<Tensor> b_bias_or_none, const Tensor b_scales,
      std::optional<Tensor> a_scales, std::optional<Tensor> global_scale,
      std::optional<Tensor> b_zeros_or_none,
      std::optional<Tensor> g_idx_or_none, std::optional<Tensor> perm_or_none,
      const Tensor workspace, const Tensor sorted_token_ids,
      const Tensor expert_ids, const Tensor num_tokens_past_padded,
      const Tensor topk_weights, const int64_t moe_block_size,
      const int64_t top_k, const bool mul_topk_weights, const int64_t b_type_id,
      const int64_t size_m, const int64_t size_n, const int64_t size_k,
      const bool is_full_k, const bool use_atomic_add,
      const bool use_fp32_reduce, const bool is_zp_float,
      const int64_t thread_k, const int64_t thread_n,
      const int64_t blocks_per_sm, Tensor out) const {
    assert(moe_block_size == moe_block_size_ && top_k == top_k_ &&
           mul_topk_weights == mul_topk_weights_ && b_type_id == b_type_id_ &&
           size_m == size_m_ && size_n == size_n_ && size_k == size_k_ &&
           is_full_k == is_full_k_ && use_atomic_add == use_atomic_add_ &&
           use_fp32_reduce == use_fp32_reduce_ && is_zp_float == is_zp_float_ &&
           thread_k == thread_k_ && thread_n == thread_n_ &&
           blocks_per_sm == blocks_per_sm_ &&
           "`MoeWna16MarlinGemm` attributes changed after descriptor "
           "creation");

    const std::equal_to<Tensor> same_metadata;
    const auto optional_matches = [&](const std::optional<Tensor>& expected,
                                      const std::optional<Tensor>& actual) {
      return expected.has_value() == actual.has_value() &&
             (!expected || same_metadata(*expected, *actual));
    };
    const auto matches =
        same_metadata(a_metadata_, a) &&
        same_metadata(b_q_weight_metadata_, b_q_weight) &&
        optional_matches(b_bias_or_none_metadata_, b_bias_or_none) &&
        same_metadata(b_scales_metadata_, b_scales) &&
        optional_matches(a_scales_metadata_, a_scales) &&
        optional_matches(global_scale_metadata_, global_scale) &&
        optional_matches(b_zeros_or_none_metadata_, b_zeros_or_none) &&
        optional_matches(g_idx_or_none_metadata_, g_idx_or_none) &&
        optional_matches(perm_or_none_metadata_, perm_or_none) &&
        same_metadata(workspace_metadata_, workspace) &&
        same_metadata(sorted_token_ids_metadata_, sorted_token_ids) &&
        same_metadata(expert_ids_metadata_, expert_ids) &&
        same_metadata(num_tokens_past_padded_metadata_,
                      num_tokens_past_padded) &&
        same_metadata(topk_weights_metadata_, topk_weights) &&
        same_metadata(out_metadata_, out);
    assert(matches &&
           "`MoeWna16MarlinGemm` tensor metadata must match descriptor");
  }

 private:
  void Validate(const Tensor a, const Tensor b_q_weight,
                std::optional<Tensor> b_bias_or_none, const Tensor b_scales,
                std::optional<Tensor> a_scales,
                std::optional<Tensor> global_scale,
                std::optional<Tensor> b_zeros_or_none,
                std::optional<Tensor> g_idx_or_none,
                std::optional<Tensor> perm_or_none, const Tensor workspace,
                const Tensor sorted_token_ids, const Tensor expert_ids,
                const Tensor num_tokens_past_padded, const Tensor topk_weights,
                const Tensor out) const {
    assert(a.ndim() == 2 && a.size(0) == size_m_ && a.size(1) == size_k_ &&
           "`MoeWna16MarlinGemm` `a` shape must match `size_m` and "
           "`size_k`");
    const auto is_a_8bit = a.dtype() == DataType::kInt8;
    const auto output_dtype = is_a_8bit ? b_scales.dtype() : a.dtype();
    assert(
        (a.dtype() == DataType::kFloat16 || a.dtype() == DataType::kBFloat16 ||
         is_a_8bit) &&
        a.IsContiguous() &&
        "`MoeWna16MarlinGemm` requires contiguous float16, bfloat16, or int8 "
        "`a`");
    assert(size_m_ > 0 && size_n_ > 0 && size_k_ > 0 && top_k_ > 0 &&
           size_k_ % 16 == 0 && size_n_ % 64 == 0 &&
           "`MoeWna16MarlinGemm` received unsupported dimensions");
    assert((moe_block_size_ == 8 ||
            (moe_block_size_ >= 16 && moe_block_size_ <= 64 &&
             moe_block_size_ % 16 == 0)) &&
           "`MoeWna16MarlinGemm` received an unsupported `moe_block_size`");
    assert(size_m_ <= std::numeric_limits<Tensor::Size>::max() / top_k_ &&
           "`MoeWna16MarlinGemm` output dimensions overflow");

    constexpr int64_t kUint4B8 = 1125899907892224;
    constexpr int64_t kUint8B128 = 1125899923621888;
    constexpr int64_t kUint4 = 1125899906843648;
    constexpr int64_t kUint8 = 1125899906844672;
    constexpr int64_t kInt4 = 1125899906908928;
    constexpr int64_t kInt8 = 1125899906909952;
    constexpr int64_t kFloat8E4M3Fn = 2814749767172868;
    constexpr int64_t kFloat4E2M1F = 562949953487106;
    const auto supported_qtype =
        b_type_id_ == kUint4B8 || b_type_id_ == kUint8B128 ||
        b_type_id_ == kUint4 || b_type_id_ == kUint8 || b_type_id_ == kInt4 ||
        b_type_id_ == kInt8 || b_type_id_ == kFloat8E4M3Fn ||
        b_type_id_ == kFloat4E2M1F;
    const auto has_zero_points = b_zeros_or_none.has_value();
    assert(supported_qtype &&
           has_zero_points == (b_type_id_ == kUint4 || b_type_id_ == kUint8) &&
           (!is_zp_float_ ||
            (has_zero_points && a.dtype() == DataType::kFloat16)) &&
           "`MoeWna16MarlinGemm` received an unsupported quantization "
           "configuration");

    const auto pack_factor =
        (b_type_id_ == kUint8B128 || b_type_id_ == kUint8 ||
         b_type_id_ == kInt8 || b_type_id_ == kFloat8E4M3Fn)
            ? 4
            : 8;
    assert(size_n_ <= std::numeric_limits<Tensor::Size>::max() / 16 &&
           b_q_weight.ndim() == 3 && b_q_weight.size(1) == size_k_ / 16 &&
           b_q_weight.size(2) == size_n_ * 16 / pack_factor &&
           b_q_weight.dtype() == DataType::kInt32 &&
           b_q_weight.IsContiguous() &&
           "`MoeWna16MarlinGemm` received invalid packed weights");
    assert(b_scales.ndim() == 3 && b_scales.size(0) == b_q_weight.size(0) &&
           b_scales.size(1) > 0 && b_scales.size(2) == size_n_ &&
           size_k_ % b_scales.size(1) == 0 &&
           (output_dtype == DataType::kFloat16 ||
            output_dtype == DataType::kBFloat16) &&
           b_scales.dtype() == output_dtype && b_scales.IsContiguous() &&
           "`MoeWna16MarlinGemm` received invalid weight scales");
    assert(
        a_scales.has_value() == is_a_8bit &&
        "`MoeWna16MarlinGemm` requires activation scales exactly for int8 `a`");
    if (a_scales) {
      assert(a_scales->shape() ==
                 Tensor::Shape({static_cast<Tensor::Size>(size_m_), 1}) &&
             a_scales->dtype() == DataType::kFloat32 &&
             a_scales->IsContiguous() &&
             "`MoeWna16MarlinGemm` received invalid activation scales");
    }

    if (b_bias_or_none) {
      assert(b_bias_or_none->ndim() == 2 &&
             b_bias_or_none->size(0) == b_q_weight.size(0) &&
             b_bias_or_none->size(1) == size_n_ &&
             b_bias_or_none->dtype() == output_dtype &&
             b_bias_or_none->IsContiguous() &&
             "`MoeWna16MarlinGemm` received invalid bias");
    }

    if (b_zeros_or_none) {
      assert(b_zeros_or_none->ndim() == 3 &&
             b_zeros_or_none->size(0) == b_q_weight.size(0) &&
             b_zeros_or_none->size(1) == b_scales.size(1) &&
             b_zeros_or_none->size(2) ==
                 (is_zp_float_ ? size_n_ : size_n_ / pack_factor) &&
             (is_zp_float_ ? b_zeros_or_none->dtype() == output_dtype
                           : b_zeros_or_none->dtype() == DataType::kInt32) &&
             "`MoeWna16MarlinGemm` received invalid zero points");
    }

    const auto same_device = [&](const Tensor tensor) {
      return tensor.device().type() == a.device().type() &&
             tensor.device().index() == a.device().index();
    };
    const auto valid_optional = [&](const std::optional<Tensor>& tensor) {
      return !tensor || (tensor->IsContiguous() && same_device(*tensor));
    };
    assert(same_device(b_q_weight) && same_device(b_scales) &&
           valid_optional(b_bias_or_none) && valid_optional(a_scales) &&
           valid_optional(global_scale) && valid_optional(b_zeros_or_none) &&
           valid_optional(g_idx_or_none) && valid_optional(perm_or_none) &&
           same_device(workspace) && same_device(sorted_token_ids) &&
           same_device(expert_ids) && same_device(num_tokens_past_padded) &&
           same_device(topk_weights) && same_device(out) &&
           "`MoeWna16MarlinGemm` requires all tensors on the input device");

    assert(g_idx_or_none.has_value() == perm_or_none.has_value() &&
           "`MoeWna16MarlinGemm` requires `g_idx_or_none` and `perm_or_none` "
           "together");
    if (g_idx_or_none) {
      assert(g_idx_or_none->ndim() > 0 && perm_or_none->ndim() > 0 &&
             g_idx_or_none->size(-1) == perm_or_none->size(-1) &&
             (g_idx_or_none->size(-1) == 0 ||
              g_idx_or_none->size(-1) == size_k_) &&
             g_idx_or_none->dtype() == DataType::kInt32 &&
             perm_or_none->dtype() == DataType::kInt32 &&
             (!is_full_k_ || b_scales.size(1) > 1) &&
             "`MoeWna16MarlinGemm` received invalid activation-order "
             "metadata");
    }

    assert(workspace.ndim() == 1 && workspace.numel() > 0 &&
           workspace.dtype() == DataType::kInt32 && workspace.IsContiguous() &&
           "`MoeWna16MarlinGemm` requires a non-empty int32 workspace");
    assert(sorted_token_ids.ndim() == 1 &&
           sorted_token_ids.dtype() == DataType::kInt32 &&
           sorted_token_ids.IsContiguous() && expert_ids.ndim() == 1 &&
           expert_ids.dtype() == DataType::kInt32 &&
           expert_ids.IsContiguous() && num_tokens_past_padded.numel() == 1 &&
           num_tokens_past_padded.dtype() == DataType::kInt32 &&
           num_tokens_past_padded.IsContiguous() &&
           "`MoeWna16MarlinGemm` received invalid routing metadata");
    assert(topk_weights.numel() == size_m_ * top_k_ &&
           ((!mul_topk_weights_ &&
             (topk_weights.dtype() == DataType::kFloat16 ||
              topk_weights.dtype() == DataType::kBFloat16)) ||
            topk_weights.dtype() == DataType::kFloat32) &&
           topk_weights.IsContiguous() &&
           "`MoeWna16MarlinGemm` received invalid top-k weights");
    assert(out.shape() ==
               Tensor::Shape({static_cast<Tensor::Size>(size_m_ * top_k_),
                              static_cast<Tensor::Size>(size_n_)}) &&
           out.dtype() == output_dtype && out.IsContiguous() &&
           "`MoeWna16MarlinGemm` output metadata is invalid");
  }

  Tensor a_metadata_;

  Tensor b_q_weight_metadata_;

  std::optional<Tensor> b_bias_or_none_metadata_;

  Tensor b_scales_metadata_;

  std::optional<Tensor> a_scales_metadata_;

  std::optional<Tensor> global_scale_metadata_;

  std::optional<Tensor> b_zeros_or_none_metadata_;

  std::optional<Tensor> g_idx_or_none_metadata_;

  std::optional<Tensor> perm_or_none_metadata_;

  Tensor workspace_metadata_;

  Tensor sorted_token_ids_metadata_;

  Tensor expert_ids_metadata_;

  Tensor num_tokens_past_padded_metadata_;

  Tensor topk_weights_metadata_;

  Tensor out_metadata_;

  int64_t moe_block_size_{0};

  int64_t top_k_{0};

  bool mul_topk_weights_{false};

  int64_t b_type_id_{0};

  int64_t size_m_{0};

  int64_t size_n_{0};

  int64_t size_k_{0};

  bool is_full_k_{false};

  bool use_atomic_add_{false};

  bool use_fp32_reduce_{false};

  bool is_zp_float_{false};

  int64_t thread_k_{0};

  int64_t thread_n_{0};

  int64_t blocks_per_sm_{0};

 protected:
  int device_index_{0};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_MOE_WNA16_MARLIN_GEMM_H_
