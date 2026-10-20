#ifndef INFINI_OPS_BASE_ELU_BACKWARD_H_
#define INFINI_OPS_BASE_ELU_BACKWARD_H_

#include "operator.h"

namespace infini::ops {

class EluBackward : public Operator<EluBackward> {
 public:
  EluBackward(const Tensor grad_output, const Tensor self_or_result,
              const double alpha, const double scale, const double input_scale,
              const bool is_result, Tensor grad_input)
      : grad_output_shape_{grad_output.shape()},
        grad_output_strides_{grad_output.strides()},
        grad_output_type_{grad_output.dtype()},
        self_or_result_shape_{self_or_result.shape()},
        self_or_result_strides_{self_or_result.strides()},
        self_or_result_type_{self_or_result.dtype()},
        grad_input_shape_{grad_input.shape()},
        grad_input_strides_{grad_input.strides()},
        grad_input_type_{grad_input.dtype()},
        alpha_{alpha},
        scale_{scale},
        input_scale_{input_scale},
        is_result_{is_result},
        device_index_{grad_input.device().index()} {}

  virtual void operator()(const Tensor grad_output, const Tensor self_or_result,
                          const double alpha, const double scale,
                          const double input_scale, const bool is_result,
                          Tensor grad_input) const = 0;

 protected:
  Tensor::Shape grad_output_shape_;

  Tensor::Strides grad_output_strides_;

  DataType grad_output_type_;

  Tensor::Shape self_or_result_shape_;

  Tensor::Strides self_or_result_strides_;

  DataType self_or_result_type_;

  Tensor::Shape grad_input_shape_;

  Tensor::Strides grad_input_strides_;

  DataType grad_input_type_;

  double alpha_{};

  double scale_{};

  double input_scale_{};

  bool is_result_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
