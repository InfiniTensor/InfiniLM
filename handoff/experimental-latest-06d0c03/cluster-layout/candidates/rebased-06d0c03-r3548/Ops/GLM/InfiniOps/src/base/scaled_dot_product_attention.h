#ifndef INFINI_OPS_BASE_SCALED_DOT_PRODUCT_ATTENTION_H_
#define INFINI_OPS_BASE_SCALED_DOT_PRODUCT_ATTENTION_H_

#include <optional>

#include "operator.h"

namespace infini::ops {

class ScaledDotProductAttention : public Operator<ScaledDotProductAttention> {
 public:
  ScaledDotProductAttention(const Tensor query, const Tensor key,
                            const Tensor value,
                            const std::optional<Tensor> attn_mask,
                            double dropout_p, bool is_causal,
                            const std::optional<double> scale, bool enable_gqa,
                            Tensor out)
      : query_shape_{query.shape()},
        key_shape_{key.shape()},
        value_shape_{value.shape()},
        attn_mask_shape_{attn_mask.has_value()
                             ? Tensor::Shape{attn_mask->shape()}
                             : Tensor::Shape{}},
        out_shape_{out.shape()},
        query_strides_{query.strides()},
        key_strides_{key.strides()},
        value_strides_{value.strides()},
        attn_mask_strides_{attn_mask.has_value()
                               ? Tensor::Strides{attn_mask->strides()}
                               : Tensor::Strides{}},
        out_strides_{out.strides()},
        query_type_{query.dtype()},
        attn_mask_type_{attn_mask.has_value() ? attn_mask->dtype()
                                              : query.dtype()},
        dropout_p_{dropout_p},
        is_causal_{is_causal},
        scale_{scale},
        enable_gqa_{enable_gqa},
        device_index_{query.device().index()} {
    assert(query.ndim() >= 2 && key.ndim() >= 2 && value.ndim() >= 2 &&
           "`ScaledDotProductAttention` requires rank-2 or higher inputs");
    assert(query.dtype() == key.dtype() && query.dtype() == value.dtype() &&
           query.dtype() == out.dtype() &&
           "`ScaledDotProductAttention` requires matching input/output "
           "dtypes");
    assert(query.size(-1) == key.size(-1) && key.size(-2) == value.size(-2) &&
           "`ScaledDotProductAttention` input dimensions are incompatible");
    Tensor::Shape expected_out_shape{query.shape()};
    expected_out_shape.back() = value.size(-1);
    assert(out.shape() == expected_out_shape &&
           "`ScaledDotProductAttention` output shape is incorrect");
    assert(dropout_p_ >= 0.0 && dropout_p_ <= 1.0 &&
           "`ScaledDotProductAttention` requires `dropout_p` in [0, 1]");
  }

  ScaledDotProductAttention(const Tensor query, const Tensor key,
                            const Tensor value, Tensor out)
      : ScaledDotProductAttention{query, key,          value, std::nullopt, 0.0,
                                  false, std::nullopt, false, out} {}

  virtual void operator()(const Tensor query, const Tensor key,
                          const Tensor value,
                          const std::optional<Tensor> attn_mask,
                          double dropout_p, bool is_causal,
                          const std::optional<double> scale, bool enable_gqa,
                          Tensor out) const = 0;

  void operator()(const Tensor query, const Tensor key, const Tensor value,
                  Tensor out) const {
    (*this)(query, key, value, std::nullopt, 0.0, false, std::nullopt, false,
            out);
  }

  template <typename TensorLike>
  static auto MakeReturnValue(
      const TensorLike& query, const TensorLike& key, const TensorLike& value,
      const std::optional<TensorLike> attn_mask = std::nullopt,
      double dropout_p = 0.0, bool is_causal = false,
      const std::optional<double> scale = std::nullopt,
      bool enable_gqa = false) {
    (void)key;
    (void)attn_mask;
    (void)dropout_p;
    (void)is_causal;
    (void)scale;
    (void)enable_gqa;

    typename TensorLike::Shape out_shape{query.shape()};
    out_shape.back() = value.size(-1);
    return TensorLike::Empty(out_shape, query.dtype(), query.device());
  }

 protected:
  Tensor::Shape query_shape_;

  Tensor::Shape key_shape_;

  Tensor::Shape value_shape_;

  Tensor::Shape attn_mask_shape_;

  Tensor::Shape out_shape_;

  Tensor::Strides query_strides_;

  Tensor::Strides key_strides_;

  Tensor::Strides value_strides_;

  Tensor::Strides attn_mask_strides_;

  Tensor::Strides out_strides_;

  DataType query_type_;

  DataType attn_mask_type_;

  double dropout_p_{0.0};

  bool is_causal_{false};

  std::optional<double> scale_;

  bool enable_gqa_{false};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
