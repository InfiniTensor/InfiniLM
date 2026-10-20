#ifndef INFINI_OPS_BASE_LINALG_NORM_H_
#define INFINI_OPS_BASE_LINALG_NORM_H_

#include <optional>
#include <string>
#include <vector>

#include "operator.h"

namespace infini::ops::linalg {

class Norm : public Operator<Norm> {
 public:
  Norm(const Tensor input, const std::optional<double> ord,
       const std::optional<std::vector<int64_t>> dim, const bool keepdim,
       const std::optional<DataType> dtype, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        ord_{ord},
        dim_{dim},
        keepdim_{keepdim},
        dtype_{dtype},
        device_index_{out.device().index()} {}

  Norm(const Tensor input, const std::string ord,
       const std::optional<std::vector<int64_t>> dim, const bool keepdim,
       const std::optional<DataType> dtype, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        dim_{dim},
        keepdim_{keepdim},
        dtype_{dtype},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input, const std::optional<double> ord,
                          const std::optional<std::vector<int64_t>> dim,
                          const bool keepdim,
                          const std::optional<DataType> dtype,
                          Tensor out) const = 0;

  virtual void operator()(const Tensor input, const std::string ord,
                          const std::optional<std::vector<int64_t>> dim,
                          const bool keepdim,
                          const std::optional<DataType> dtype,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  std::optional<double> ord_{};

  std::optional<std::vector<int64_t>> dim_{};

  bool keepdim_{};

  std::optional<DataType> dtype_{};

  int device_index_{0};
};

}  // namespace infini::ops::linalg

#endif
