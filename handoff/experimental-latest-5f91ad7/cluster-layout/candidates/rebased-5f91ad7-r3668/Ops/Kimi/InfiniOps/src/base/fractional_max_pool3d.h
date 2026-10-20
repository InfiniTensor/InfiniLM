#ifndef INFINI_OPS_BASE_FRACTIONAL_MAX_POOL3D_H_
#define INFINI_OPS_BASE_FRACTIONAL_MAX_POOL3D_H_

#include <vector>

#include "operator.h"

namespace infini::ops {

class FractionalMaxPool3d : public Operator<FractionalMaxPool3d> {
 public:
  FractionalMaxPool3d(const Tensor input, const Tensor random_samples,
                      const std::vector<int64_t> kernel_size,
                      const std::vector<int64_t> output_size, Tensor output,
                      Tensor indices)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        random_samples_shape_{random_samples.shape()},
        random_samples_strides_{random_samples.strides()},
        random_samples_type_{random_samples.dtype()},
        output_shape_{output.shape()},
        output_strides_{output.strides()},
        output_type_{output.dtype()},
        indices_shape_{indices.shape()},
        indices_strides_{indices.strides()},
        indices_type_{indices.dtype()},
        kernel_size_{kernel_size},
        output_size_{output_size},
        device_index_{output.device().index()} {}

  virtual void operator()(const Tensor input, const Tensor random_samples,
                          const std::vector<int64_t> kernel_size,
                          const std::vector<int64_t> output_size, Tensor output,
                          Tensor indices) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape random_samples_shape_;

  Tensor::Strides random_samples_strides_;

  DataType random_samples_type_;

  Tensor::Shape output_shape_;

  Tensor::Strides output_strides_;

  DataType output_type_;

  Tensor::Shape indices_shape_;

  Tensor::Strides indices_strides_;

  DataType indices_type_;

  std::vector<int64_t> kernel_size_{};

  std::vector<int64_t> output_size_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
