#ifndef INFINI_OPS_BASE_CAT_H_
#define INFINI_OPS_BASE_CAT_H_

#include <vector>

#include "operator.h"

namespace infini::ops {

class Cat : public Operator<Cat> {
 public:
  static bool supports(const std::vector<Tensor>& tensors, int64_t dim,
                       Tensor out) {
    if (tensors.empty() || dim < -static_cast<int64_t>(out.ndim()) ||
        dim >= static_cast<int64_t>(out.ndim()))
      return false;
    dim = dim < 0 ? dim + static_cast<int64_t>(out.ndim()) : dim;
    Tensor::Size total = 0;
    for (const auto& tensor : tensors) {
      if (tensor.ndim() != out.ndim() || tensor.dtype() != out.dtype())
        return false;
      for (Tensor::Size axis = 0; axis < out.ndim(); ++axis)
        if (axis != static_cast<Tensor::Size>(dim) &&
            tensor.size(axis) != out.size(axis))
          return false;
      total += tensor.size(dim);
    }
    return total == out.size(dim);
  }

  Cat(const std::vector<Tensor> tensors, const int64_t dim, Tensor out)
      : out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        input_count_{tensors.size()} {
    assert(!tensors.empty() && "`Cat` requires a non-empty tensor list");

    auto ndim = static_cast<int64_t>(out.ndim());
    dim_ = dim < 0 ? dim + ndim : dim;
    assert(dim_ >= 0 && dim_ < ndim && "`Cat` dim out of range");

    Tensor::Size cat_size = 0;
    for (const auto& tensor : tensors) {
      assert(tensor.ndim() == out.ndim() &&
             "`Cat` requires all tensors to have the output rank");
      assert(tensor.dtype() == out.dtype() &&
             "`Cat` requires all tensors to have the output dtype");

      for (Tensor::Size axis = 0; axis < out.ndim(); ++axis) {
        if (axis != static_cast<Tensor::Size>(dim_)) {
          assert(tensor.size(axis) == out.size(axis) &&
                 "`Cat` input dimensions must match outside `dim`");
        }
      }
      cat_size += tensor.size(dim_);
    }
    assert(cat_size == out.size(dim_) &&
           "`Cat` output size along `dim` must equal the input sum");
  }

  virtual void operator()(const std::vector<Tensor> tensors, const int64_t dim,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  int64_t dim_;

  size_t input_count_;
};

}  // namespace infini::ops

#endif
