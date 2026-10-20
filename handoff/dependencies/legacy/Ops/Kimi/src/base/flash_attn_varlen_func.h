#ifndef INFINI_OPS_BASE_FLASH_ATTN_VARLEN_FUNC_H_
#define INFINI_OPS_BASE_FLASH_ATTN_VARLEN_FUNC_H_

#include <cassert>
#include <cstdint>
#include <optional>
#include <vector>

#include "operator.h"

namespace infini::ops {

// Packed variable-length attention aligned with Dao-AILab FlashAttention's
// `flash_attn_varlen_func` public interface.
class FlashAttnVarlenFunc : public Operator<FlashAttnVarlenFunc> {
 public:
  static constexpr bool kAutomaticDefault = true;

  // Common metadata, not native layout/capacity or capture qualification.
  // Planning may use null data; invocation checks actual nonempty storage.
  static bool supports(
      const Tensor q, const Tensor k, const Tensor v, const Tensor cu_seqlens_q,
      const Tensor cu_seqlens_k, const std::optional<Tensor> alibi_slopes,
      const std::optional<Tensor> block_table, const int64_t max_seqlen_q,
      const int64_t max_seqlen_k, const double dropout_p,
      const std::optional<double> softmax_scale, const bool causal,
      const std::vector<int64_t> window_size, const double softcap,
      const bool deterministic, const bool return_attn_probs, Tensor out,
      std::optional<Tensor> softmax_lse, std::optional<Tensor> s_dmask) {
    if (q.ndim() != 3 || k.ndim() != (block_table ? 4 : 3) ||
        v.ndim() != k.ndim() || k.shape() != v.shape() || out.shape() != q.shape() ||
        (q.dtype() != DataType::kFloat16 && q.dtype() != DataType::kBFloat16) ||
        k.dtype() != q.dtype() || v.dtype() != q.dtype() || out.dtype() != q.dtype() ||
        cu_seqlens_q.ndim() != 1 || cu_seqlens_k.ndim() != 1 ||
        cu_seqlens_q.shape() != cu_seqlens_k.shape() || cu_seqlens_q.size(0) < 2 ||
        cu_seqlens_q.dtype() != DataType::kInt32 ||
        cu_seqlens_k.dtype() != DataType::kInt32 ||
        (alibi_slopes && ((alibi_slopes->ndim() != 1 && alibi_slopes->ndim() != 2) ||
                         alibi_slopes->dtype() != DataType::kFloat32)) ||
        (block_table && (block_table->ndim() != 2 || block_table->dtype() != DataType::kInt32)) ||
        softmax_lse.has_value() != s_dmask.has_value() ||
        return_attn_probs != softmax_lse.has_value() ||
        (softmax_lse && (softmax_lse->dtype() != DataType::kFloat32 || s_dmask->dtype() != q.dtype())) ||
        max_seqlen_q <= 0 || max_seqlen_k <= 0 ||
        window_size.size() != 2 || window_size[0] < -1 || window_size[1] < -1 ||
        dropout_p != 0.0 || softcap != 0.0 || deterministic) return false;
    const auto metadata = [&](const Tensor& tensor) {
      return tensor.device().type() == q.device().type() &&
             tensor.device().index() == q.device().index();
    };
    for (const auto& tensor : {q, k, v, cu_seqlens_q, cu_seqlens_k, out})
      if (!metadata(tensor)) return false;
    for (const auto& tensor : {alibi_slopes, block_table, softmax_lse, s_dmask})
      if (tensor && !metadata(*tensor)) return false;
    // Span validation above establishes stride cardinality before axis access.
    if (q.size(1) == 0 || k.size(-2) == 0 || q.size(1) % k.size(-2) != 0 ||
        q.size(2) == 0 || q.size(2) > 256 || q.size(2) % 8 != 0 ||
        q.size(2) != k.size(-1) || q.stride(-1) != 1 || k.stride(-1) != 1 ||
        v.stride(-1) != 1 || out.stride(-1) != 1 ||
        !cu_seqlens_q.IsContiguous() || !cu_seqlens_k.IsContiguous()) return false;
    if (softmax_lse &&
        (softmax_lse->shape() != Tensor::Shape{q.size(1), q.size(0)} ||
         s_dmask->shape() != Tensor::Shape{0})) return false;
    if (block_table &&
        (block_table->size(0) != cu_seqlens_q.size(0) - 1 ||
         !block_table->IsContiguous() || k.size(1) % 256 != 0)) return false;
    if (alibi_slopes &&
        (!alibi_slopes->IsContiguous() ||
         !((alibi_slopes->ndim() == 1 && alibi_slopes->size(0) == q.size(1)) ||
           (alibi_slopes->ndim() == 2 &&
            alibi_slopes->size(0) == cu_seqlens_q.size(0) - 1 &&
            alibi_slopes->size(1) == q.size(1))))) return false;
    (void)softmax_scale;
    (void)causal;
    return true;
  }

  static bool supports(const Tensor q, const Tensor k, const Tensor v,
                       const Tensor cu_seqlens_q, const Tensor cu_seqlens_k,
                       const int64_t max_seqlen_q, const int64_t max_seqlen_k,
                       Tensor out) {
    return supports(q, k, v, cu_seqlens_q, cu_seqlens_k, std::nullopt,
                    std::nullopt, max_seqlen_q, max_seqlen_k, 0.0, std::nullopt,
                    false, {-1, -1}, 0.0, false, false, out, std::nullopt,
                    std::nullopt);
  }

  FlashAttnVarlenFunc(const Tensor q, const Tensor k, const Tensor v,
                      const Tensor cu_seqlens_q, const Tensor cu_seqlens_k,
                      const int64_t max_seqlen_q, const int64_t max_seqlen_k,
                      Tensor out)
      : FlashAttnVarlenFunc{q,
                            k,
                            v,
                            cu_seqlens_q,
                            cu_seqlens_k,
                            std::nullopt,
                            std::nullopt,
                            max_seqlen_q,
                            max_seqlen_k,
                            0.0,
                            std::nullopt,
                            false,
                            {-1, -1},
                            0.0,
                            false,
                            false,
                            out,
                            std::nullopt,
                            std::nullopt} {}

  FlashAttnVarlenFunc(
      const Tensor q, const Tensor k, const Tensor v, const Tensor cu_seqlens_q,
      const Tensor cu_seqlens_k, const std::optional<Tensor> alibi_slopes,
      const std::optional<Tensor> block_table, const int64_t max_seqlen_q,
      const int64_t max_seqlen_k, const double dropout_p,
      const std::optional<double> softmax_scale, const bool causal,
      const std::vector<int64_t> window_size, const double softcap,
      const bool deterministic, const bool return_attn_probs, Tensor out,
      std::optional<Tensor> softmax_lse, std::optional<Tensor> s_dmask)
      : q_shape_{q.shape()},
        k_shape_{k.shape()},
        v_shape_{v.shape()},
        cu_seqlens_q_shape_{cu_seqlens_q.shape()},
        cu_seqlens_k_shape_{cu_seqlens_k.shape()},
        alibi_slopes_shape_{alibi_slopes.has_value()
                                ? Tensor::Shape{alibi_slopes->shape()}
                                : Tensor::Shape{}},
        block_table_shape_{block_table.has_value()
                               ? Tensor::Shape{block_table->shape()}
                               : Tensor::Shape{}},
        out_shape_{out.shape()},
        softmax_lse_shape_{softmax_lse.has_value()
                               ? Tensor::Shape{softmax_lse->shape()}
                               : Tensor::Shape{}},
        s_dmask_shape_{s_dmask.has_value() ? Tensor::Shape{s_dmask->shape()}
                                           : Tensor::Shape{}},
        q_strides_{q.strides()},
        k_strides_{k.strides()},
        v_strides_{v.strides()},
        cu_seqlens_q_strides_{cu_seqlens_q.strides()},
        cu_seqlens_k_strides_{cu_seqlens_k.strides()},
        alibi_slopes_strides_{alibi_slopes.has_value()
                                  ? Tensor::Strides{alibi_slopes->strides()}
                                  : Tensor::Strides{}},
        block_table_strides_{block_table.has_value()
                                 ? Tensor::Strides{block_table->strides()}
                                 : Tensor::Strides{}},
        out_strides_{out.strides()},
        softmax_lse_strides_{softmax_lse.has_value()
                                 ? Tensor::Strides{softmax_lse->strides()}
                                 : Tensor::Strides{}},
        s_dmask_strides_{s_dmask.has_value()
                             ? Tensor::Strides{s_dmask->strides()}
                             : Tensor::Strides{}},
        q_dtype_{q.dtype()},
        k_dtype_{k.dtype()},
        v_dtype_{v.dtype()},
        cu_seqlens_q_dtype_{cu_seqlens_q.dtype()},
        cu_seqlens_k_dtype_{cu_seqlens_k.dtype()},
        alibi_slopes_dtype_{alibi_slopes.has_value() ? alibi_slopes->dtype()
                                                     : DataType::kFloat32},
        block_table_dtype_{block_table.has_value() ? block_table->dtype()
                                                   : DataType::kInt32},
        out_dtype_{out.dtype()},
        softmax_lse_dtype_{softmax_lse.has_value() ? softmax_lse->dtype()
                                                   : DataType::kFloat32},
        s_dmask_dtype_{s_dmask.has_value() ? s_dmask->dtype() : q.dtype()},
        has_auxiliary_outputs_{softmax_lse.has_value() && s_dmask.has_value()},
        device_index_{q.device().index()} {
    assert(q.ndim() == 3 &&
           ((!block_table.has_value() && k.ndim() == 3 && v.ndim() == 3) ||
            (block_table.has_value() && k.ndim() == 4 && v.ndim() == 4)) &&
           "`FlashAttnVarlenFunc` requires packed 3D Q and either packed 3D "
           "or paged 4D K and V tensors");
    assert(k.shape() == v.shape() &&
           "`FlashAttnVarlenFunc` requires K and V to have the same shape");
    assert(q.size(1) > 0 && k.size(-2) > 0 && q.size(2) == k.size(-1) &&
           q.size(1) % k.size(-2) == 0 &&
           "`FlashAttnVarlenFunc` requires compatible Q and KV heads");
    assert(q.size(2) > 0 && q.size(2) <= 256 && q.size(2) % 8 == 0 &&
           "`FlashAttnVarlenFunc` requires a head dimension divisible by 8 "
           "and no greater than 256");
    assert(out.shape() == q.shape() &&
           "`FlashAttnVarlenFunc` output must have the same shape as Q");
    assert(softmax_lse.has_value() == s_dmask.has_value() &&
           "`FlashAttnVarlenFunc` auxiliary outputs must be provided "
           "together");
    assert(return_attn_probs == has_auxiliary_outputs_ &&
           "`FlashAttnVarlenFunc` requires auxiliary outputs exactly when "
           "`return_attn_probs` is true");
    if (has_auxiliary_outputs_) {
      assert((softmax_lse->shape() == Tensor::Shape{q.size(1), q.size(0)} &&
              softmax_lse_dtype_ == DataType::kFloat32 &&
              "`FlashAttnVarlenFunc` softmax LSE output must have shape "
              "(num_heads, total_q) and dtype float32"));
      assert(s_dmask->shape() == Tensor::Shape{0} &&
             s_dmask_dtype_ == q_dtype_ &&
             "`FlashAttnVarlenFunc` inference attention mask output must be "
             "empty and match the Q dtype");
    }
    assert(
        (q_dtype_ == DataType::kFloat16 || q_dtype_ == DataType::kBFloat16) &&
        q_dtype_ == k_dtype_ && q_dtype_ == v_dtype_ &&
        q_dtype_ == out_dtype_ &&
        "`FlashAttnVarlenFunc` requires matching float16 or bfloat16 Q, "
        "K, V, and output dtypes");
    assert(q.stride(-1) == 1 && k.stride(-1) == 1 && v.stride(-1) == 1 &&
           out.stride(-1) == 1 &&
           "`FlashAttnVarlenFunc` requires contiguous head dimensions");
    assert(cu_seqlens_q.ndim() == 1 && cu_seqlens_k.ndim() == 1 &&
           cu_seqlens_q.shape() == cu_seqlens_k.shape() &&
           cu_seqlens_q.numel() >= 2 &&
           "`FlashAttnVarlenFunc` cumulative sequence tensors must be "
           "matching non-empty vectors");
    assert(cu_seqlens_q_dtype_ == DataType::kInt32 &&
           cu_seqlens_k_dtype_ == DataType::kInt32 &&
           cu_seqlens_q.IsContiguous() && cu_seqlens_k.IsContiguous() &&
           "`FlashAttnVarlenFunc` cumulative sequence tensors must be "
           "contiguous int32 tensors");
    assert(max_seqlen_q > 0 && max_seqlen_k > 0 &&
           "`FlashAttnVarlenFunc` maximum sequence lengths must be positive");
    assert(window_size.size() == 2 && window_size[0] >= -1 &&
           window_size[1] >= -1 &&
           "`FlashAttnVarlenFunc` `window_size` must contain two values >= -1");

    assert(dropout_p == 0.0 &&
           "`FlashAttnVarlenFunc` initially supports inference only");
    assert(softcap == 0.0 &&
           "`FlashAttnVarlenFunc` does not yet support softcap");
    assert(!deterministic &&
           "`FlashAttnVarlenFunc` does not yet support deterministic mode");
    if (block_table.has_value()) {
      assert(block_table->ndim() == 2 &&
             block_table->size(0) + 1 == cu_seqlens_q.size(0) &&
             block_table_dtype_ == DataType::kInt32 &&
             block_table->IsContiguous() && k.size(1) % 256 == 0 &&
             "`FlashAttnVarlenFunc` requires a contiguous int32 block table "
             "and page size divisible by 256");
    }
    if (alibi_slopes.has_value()) {
      assert(
          (alibi_slopes->ndim() == 1 || alibi_slopes->ndim() == 2) &&
          alibi_slopes_dtype_ == DataType::kFloat32 &&
          alibi_slopes->IsContiguous() &&
          ((alibi_slopes->ndim() == 1 && alibi_slopes->size(0) == q.size(1)) ||
           (alibi_slopes->ndim() == 2 &&
            alibi_slopes->size(0) + 1 == cu_seqlens_q.size(0) &&
            alibi_slopes->size(1) == q.size(1))) &&
          "`FlashAttnVarlenFunc` received incompatible ALiBi slopes");
    }

    const auto same_device_as_q = [&](const Tensor tensor) {
      return tensor.device().type() == q.device().type() &&
             tensor.device().index() == q.device().index();
    };
    assert(same_device_as_q(k) && same_device_as_q(v) &&
           same_device_as_q(cu_seqlens_q) && same_device_as_q(cu_seqlens_k) &&
           same_device_as_q(out) &&
           (!alibi_slopes.has_value() || same_device_as_q(*alibi_slopes)) &&
           (!block_table.has_value() || same_device_as_q(*block_table)) &&
           (!softmax_lse.has_value() || same_device_as_q(*softmax_lse)) &&
           (!s_dmask.has_value() || same_device_as_q(*s_dmask)) &&
           "`FlashAttnVarlenFunc` tensors must be on the same device");

    (void)softmax_scale;
    (void)causal;
  }

  void operator()(const Tensor q, const Tensor k, const Tensor v,
                  const Tensor cu_seqlens_q, const Tensor cu_seqlens_k,
                  const int64_t max_seqlen_q, const int64_t max_seqlen_k,
                  Tensor out) const {
    (*this)(q, k, v, cu_seqlens_q, cu_seqlens_k, std::nullopt, std::nullopt,
            max_seqlen_q, max_seqlen_k, 0.0, std::nullopt, false, {-1, -1}, 0.0,
            false, false, out, std::nullopt, std::nullopt);
  }

  virtual void operator()(
      const Tensor q, const Tensor k, const Tensor v, const Tensor cu_seqlens_q,
      const Tensor cu_seqlens_k, const std::optional<Tensor> alibi_slopes,
      const std::optional<Tensor> block_table, const int64_t max_seqlen_q,
      const int64_t max_seqlen_k, const double dropout_p,
      const std::optional<double> softmax_scale, const bool causal,
      const std::vector<int64_t> window_size, const double softcap,
      const bool deterministic, const bool return_attn_probs, Tensor out,
      std::optional<Tensor> softmax_lse,
      std::optional<Tensor> s_dmask) const = 0;

 protected:
  Tensor::Shape q_shape_;

  Tensor::Shape k_shape_;

  Tensor::Shape v_shape_;

  Tensor::Shape cu_seqlens_q_shape_;

  Tensor::Shape cu_seqlens_k_shape_;

  Tensor::Shape alibi_slopes_shape_;

  Tensor::Shape block_table_shape_;

  Tensor::Shape out_shape_;

  Tensor::Shape softmax_lse_shape_;

  Tensor::Shape s_dmask_shape_;

  Tensor::Strides q_strides_;

  Tensor::Strides k_strides_;

  Tensor::Strides v_strides_;

  Tensor::Strides cu_seqlens_q_strides_;

  Tensor::Strides cu_seqlens_k_strides_;

  Tensor::Strides alibi_slopes_strides_;

  Tensor::Strides block_table_strides_;

  Tensor::Strides out_strides_;

  Tensor::Strides softmax_lse_strides_;

  Tensor::Strides s_dmask_strides_;

  DataType q_dtype_;

  DataType k_dtype_;

  DataType v_dtype_;

  DataType cu_seqlens_q_dtype_;

  DataType cu_seqlens_k_dtype_;

  DataType alibi_slopes_dtype_;

  DataType block_table_dtype_;

  DataType out_dtype_;

  DataType softmax_lse_dtype_;

  DataType s_dmask_dtype_;

  bool has_auxiliary_outputs_{false};

  int device_index_{0};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_FLASH_ATTN_VARLEN_FUNC_H_
