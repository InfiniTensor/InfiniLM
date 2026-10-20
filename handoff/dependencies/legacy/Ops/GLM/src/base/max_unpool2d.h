#ifndef INFINI_OPS_BASE_MAX_UNPOOL2D_H_
#define INFINI_OPS_BASE_MAX_UNPOOL2D_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "common/op_utils/max_unpool.h"

namespace infini::ops {

class MaxUnpool2d : public Operator<MaxUnpool2d> {
 public:
  MaxUnpool2d(const Tensor input, const Tensor indices,
              const std::vector<int64_t> kernel_size,
              const std::optional<std::vector<int64_t>> stride,
              const std::vector<int64_t> padding,
              const std::optional<std::vector<int64_t>> output_size, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        indices_shape_{indices.shape()},
        indices_strides_{indices.strides()},
        indices_type_{indices.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        output_size_{},
        device_index_{out.device().index()} {
    auto geometry = max_unpool_detail::ResolveGeometry<2>(
        input, kernel_size, stride, padding, output_size);
    output_size_ = std::move(geometry.first);
  }

  /// \deprecated Use the overload that accepts `kernel_size`, `stride`,
  /// `padding`, and `output_size` instead.
  [[deprecated("Use the `kernel_size` overload instead.")]]
  MaxUnpool2d(const Tensor input, const Tensor indices,
              const std::vector<int64_t> output_size, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        indices_shape_{indices.shape()},
        indices_strides_{indices.strides()},
        indices_type_{indices.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        output_size_{output_size},
        device_index_{out.device().index()} {}

  void operator()(const Tensor input, const Tensor indices,
                  const std::vector<int64_t> kernel_size,
                  const std::optional<std::vector<int64_t>> stride,
                  const std::vector<int64_t> padding,
                  const std::optional<std::vector<int64_t>> output_size,
                  Tensor out) const {
    auto geometry = max_unpool_detail::ResolveGeometry<2>(
        input, kernel_size, stride, padding, output_size);
    (*this)(input, indices, std::move(geometry.first), out);
  }

  /// \deprecated Use the overload that accepts `kernel_size`, `stride`,
  /// `padding`, and `output_size` instead.
  [[deprecated("Use the `kernel_size` overload instead.")]] virtual void
  operator()(const Tensor input, const Tensor indices,
             const std::vector<int64_t> output_size, Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape indices_shape_;

  Tensor::Strides indices_strides_;

  DataType indices_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  std::vector<int64_t> output_size_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
