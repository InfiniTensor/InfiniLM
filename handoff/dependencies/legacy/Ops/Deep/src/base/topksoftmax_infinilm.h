#ifndef INFINI_OPS_BASE_TOPKSOFTMAX_INFINILM_H_
#define INFINI_OPS_BASE_TOPKSOFTMAX_INFINILM_H_

#include <cassert>

#include "operator.h"

namespace infini::ops {

/// \deprecated Use `TopkSoftmax`. This interface will be removed in a future
/// release.
class [[deprecated("Use `TopkSoftmax` instead.")]] TopksoftmaxInfinilm
    : public Operator<TopksoftmaxInfinilm> {
 public:
  TopksoftmaxInfinilm(const Tensor input, const int64_t topk, const bool norm,
                      Tensor values, Tensor indices)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        values_shape_{values.shape()},
        values_strides_{values.strides()},
        values_type_{values.dtype()},
        indices_shape_{indices.shape()},
        indices_strides_{indices.strides()},
        indices_type_{indices.dtype()},
        topk_{topk},
        norm_{norm},
        row_count_{input.size(0)},
        width_{input.size(1)},
        device_index_{values.device().index()} {
    assert(input.ndim() == 2 &&
           "`TopksoftmaxInfinilm` input must be a 2D tensor");
    assert(topk_ > 0 && topk_ <= static_cast<int64_t>(width_) &&
           "`TopksoftmaxInfinilm` topk must be in (0, input.size(1)]");
    assert(values_shape_ == indices_shape_ &&
           "`TopksoftmaxInfinilm` values and indices shapes must match");
    assert(
        values_shape_.size() == 2 && values_shape_[0] == row_count_ &&
        values_shape_[1] == static_cast<Tensor::Size>(topk_) &&
        "`TopksoftmaxInfinilm` outputs must have shape (input.size(0), topk)");
    assert(values_type_ == DataType::kFloat32 &&
           "`TopksoftmaxInfinilm` values output must be float32");
    assert(indices_type_ == DataType::kInt32 &&
           "`TopksoftmaxInfinilm` indices output must be int32");
    assert((input_type_ == DataType::kFloat16 ||
            input_type_ == DataType::kBFloat16 ||
            input_type_ == DataType::kFloat32 ||
            input_type_ == DataType::kFloat64) &&
           "`TopksoftmaxInfinilm` input must be a floating point tensor");
    assert(
        !values.HasBroadcastDim() && !indices.HasBroadcastDim() &&
        "`TopksoftmaxInfinilm` outputs must not have broadcasted dimensions");
  }

  virtual void operator()(const Tensor input, const int64_t topk,
                          const bool norm, Tensor values,
                          Tensor indices) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape values_shape_;

  Tensor::Strides values_strides_;

  DataType values_type_;

  Tensor::Shape indices_shape_;

  Tensor::Strides indices_strides_;

  DataType indices_type_;

  int64_t topk_{0};

  bool norm_{false};

  Tensor::Size row_count_{0};

  Tensor::Size width_{0};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
