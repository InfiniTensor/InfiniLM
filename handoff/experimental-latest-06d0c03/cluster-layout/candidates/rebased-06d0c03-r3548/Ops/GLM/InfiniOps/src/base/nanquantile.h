#ifndef INFINI_OPS_BASE_NANQUANTILE_H_
#define INFINI_OPS_BASE_NANQUANTILE_H_

#include <optional>
#include <string>

#include "operator.h"

namespace infini::ops {

class Nanquantile : public Operator<Nanquantile> {
 public:
  Nanquantile(const Tensor input, const Tensor q,
              const std::optional<int64_t> dim, const bool keepdim,
              const std::string interpolation, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        q_shape_{q.shape()},
        q_strides_{q.strides()},
        q_type_{q.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        dim_{dim},
        keepdim_{keepdim},
        interpolation_{interpolation},
        device_index_{out.device().index()} {}

  Nanquantile(const Tensor input, const double q,
              const std::optional<int64_t> dim, const bool keepdim,
              const std::string interpolation, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        dim_{dim},
        keepdim_{keepdim},
        interpolation_{interpolation},
        q_{q},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input, const Tensor q,
                          const std::optional<int64_t> dim, const bool keepdim,
                          const std::string interpolation,
                          Tensor out) const = 0;

  virtual void operator()(const Tensor input, const double q,
                          const std::optional<int64_t> dim, const bool keepdim,
                          const std::string interpolation,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape q_shape_;

  Tensor::Strides q_strides_;

  DataType q_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  std::optional<int64_t> dim_{};

  bool keepdim_{};

  std::string interpolation_{};

  double q_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
