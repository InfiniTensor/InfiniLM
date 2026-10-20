#ifndef INFINI_OPS_BASE_TENSORDOT_H_
#define INFINI_OPS_BASE_TENSORDOT_H_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "operator.h"

namespace infini::ops {

class Tensordot : public Operator<Tensordot> {
 public:
  Tensordot(const Tensor a, const Tensor b, const int64_t dims, Tensor out)
      : Tensordot{a, b, ExpandDims(a, b, dims), out} {}

  Tensordot(const Tensor a, const Tensor b,
            const std::vector<std::vector<int64_t>> dims, Tensor out)
      : input_shape_{a.shape()},
        input_strides_{a.strides()},
        input_type_{a.dtype()},
        other_shape_{b.shape()},
        other_strides_{b.strides()},
        other_type_{b.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        dims_self_{DimGroup(dims, 0)},
        dims_other_{DimGroup(dims, 1)},
        device_index_{out.device().index()} {}

  /// \deprecated Use an overload taking a single `dims` argument instead.
  [[deprecated("Use an overload taking a single `dims` argument instead.")]]
  Tensordot(const Tensor input, const Tensor other,
            const std::vector<int64_t> dims_self,
            const std::vector<int64_t> dims_other, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        other_shape_{other.shape()},
        other_strides_{other.strides()},
        other_type_{other.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        dims_self_{dims_self},
        dims_other_{dims_other},
        device_index_{out.device().index()} {}

  void operator()(const Tensor a, const Tensor b, const int64_t dims,
                  Tensor out) const {
    const auto expanded_dims = ExpandDims(a, b, dims);
    (*this)(a, b, expanded_dims[0], expanded_dims[1], out);
  }

  void operator()(const Tensor a, const Tensor b,
                  const std::vector<std::vector<int64_t>> dims,
                  Tensor out) const {
    (*this)(a, b, DimGroup(dims, 0), DimGroup(dims, 1), out);
  }

  /// \deprecated Use an overload taking a single `dims` argument instead.
  [[deprecated("Use an overload taking a single `dims` argument instead.")]]
  virtual void operator()(const Tensor input, const Tensor other,
                          const std::vector<int64_t> dims_self,
                          const std::vector<int64_t> dims_other,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape other_shape_;

  Tensor::Strides other_strides_;

  DataType other_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  std::vector<int64_t> dims_self_{};

  std::vector<int64_t> dims_other_{};

  int device_index_{0};

 private:
  static std::vector<std::vector<int64_t>> ExpandDims(const Tensor a,
                                                      const Tensor b,
                                                      const int64_t dims) {
    assert(dims >= 0 && dims <= static_cast<int64_t>(a.ndim()) &&
           dims <= static_cast<int64_t>(b.ndim()) &&
           "`Tensordot` expects non-negative `dims` no greater than either "
           "input rank");

    std::vector<std::vector<int64_t>> expanded_dims(2);
    for (int64_t dim = 0; dim < dims; ++dim) {
      expanded_dims[0].push_back(dim - dims);
      expanded_dims[1].push_back(dim);
    }

    return expanded_dims;
  }

  static const std::vector<int64_t>& DimGroup(
      const std::vector<std::vector<int64_t>>& dims, const std::size_t index) {
    assert(dims.size() == 2 &&
           "`Tensordot` expects `dims` to contain exactly two dimension "
           "groups");
    return dims[index];
  }
};

}  // namespace infini::ops

#endif
