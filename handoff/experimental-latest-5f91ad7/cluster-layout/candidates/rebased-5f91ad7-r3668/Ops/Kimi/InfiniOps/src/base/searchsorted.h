#ifndef INFINI_OPS_BASE_SEARCHSORTED_H_
#define INFINI_OPS_BASE_SEARCHSORTED_H_

#include <optional>
#include <string>

#include "operator.h"

namespace infini::ops {

class Searchsorted : public Operator<Searchsorted> {
 public:
  Searchsorted(const Tensor sorted_sequence, const Tensor input,
               const std::optional<Tensor> sorter, const bool out_int32,
               const bool right, const std::optional<std::string> side,
               Tensor out)
      : sorted_sequence_shape_{sorted_sequence.shape()},
        sorted_sequence_strides_{sorted_sequence.strides()},
        sorted_sequence_type_{sorted_sequence.dtype()},
        input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        has_sorter_{sorter.has_value()},
        sorter_shape_{sorter ? Tensor::Shape{sorter->shape()}
                             : Tensor::Shape{}},
        sorter_strides_{sorter ? Tensor::Strides{sorter->strides()}
                               : Tensor::Strides{}},
        sorter_type_{sorter ? sorter->dtype() : DataType::kFloat32},
        out_int32_{out_int32},
        right_{right},
        side_{side},
        device_index_{out.device().index()} {}

  /// \deprecated Use `(sorted_sequence, input, sorter, out_int32, right,
  /// side, out)`. This constructor will be removed in a future release.
  [[deprecated("Place the optional `sorter` Tensor before attributes.")]]
  Searchsorted(const Tensor sorted_sequence, const Tensor input,
               const bool out_int32, const bool right,
               const std::optional<std::string> side,
               const std::optional<Tensor> sorter, Tensor out)
      : Searchsorted{sorted_sequence, input, sorter, out_int32,
                     right,           side,  out} {}

  Searchsorted(const Tensor sorted_sequence, const double input,
               const std::optional<Tensor> sorter, const bool out_int32,
               const bool right, const std::optional<std::string> side,
               Tensor out)
      : sorted_sequence_shape_{sorted_sequence.shape()},
        sorted_sequence_strides_{sorted_sequence.strides()},
        sorted_sequence_type_{sorted_sequence.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        has_sorter_{sorter.has_value()},
        sorter_shape_{sorter ? Tensor::Shape{sorter->shape()}
                             : Tensor::Shape{}},
        sorter_strides_{sorter ? Tensor::Strides{sorter->strides()}
                               : Tensor::Strides{}},
        sorter_type_{sorter ? sorter->dtype() : DataType::kFloat32},
        out_int32_{out_int32},
        right_{right},
        side_{side},
        input_{input},
        device_index_{out.device().index()} {}

  /// \deprecated Use `(sorted_sequence, input, sorter, out_int32, right,
  /// side, out)`. This constructor will be removed in a future release.
  [[deprecated("Place the optional `sorter` Tensor before attributes.")]]
  Searchsorted(const Tensor sorted_sequence, const double input,
               const bool out_int32, const bool right,
               const std::optional<std::string> side,
               const std::optional<Tensor> sorter, Tensor out)
      : Searchsorted{sorted_sequence, input, sorter, out_int32,
                     right,           side,  out} {}

  void operator()(const Tensor sorted_sequence, const Tensor input,
                  const std::optional<Tensor> sorter, const bool out_int32,
                  const bool right, const std::optional<std::string> side,
                  Tensor out) const {
    (*this)(sorted_sequence, input, out_int32, right, side, sorter, out);
  }

  void operator()(const Tensor sorted_sequence, const double input,
                  const std::optional<Tensor> sorter, const bool out_int32,
                  const bool right, const std::optional<std::string> side,
                  Tensor out) const {
    (*this)(sorted_sequence, input, out_int32, right, side, sorter, out);
  }

  /// \deprecated Place `sorter` before attributes. This overload will be
  /// removed in a future release.
  [[deprecated("Place the optional `sorter` Tensor before attributes.")]]
  virtual void operator()(const Tensor sorted_sequence, const Tensor input,
                          const bool out_int32, const bool right,
                          const std::optional<std::string> side,
                          const std::optional<Tensor> sorter,
                          Tensor out) const = 0;

  /// \deprecated Place `sorter` before attributes. This overload will be
  /// removed in a future release.
  [[deprecated("Place the optional `sorter` Tensor before attributes.")]]
  virtual void operator()(const Tensor sorted_sequence, const double input,
                          const bool out_int32, const bool right,
                          const std::optional<std::string> side,
                          const std::optional<Tensor> sorter,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape sorted_sequence_shape_;

  Tensor::Strides sorted_sequence_strides_;

  DataType sorted_sequence_type_;

  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  bool has_sorter_{false};

  Tensor::Shape sorter_shape_;

  Tensor::Strides sorter_strides_;

  DataType sorter_type_{DataType::kFloat32};

  bool out_int32_{};

  bool right_{};

  std::optional<std::string> side_{};

  double input_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
