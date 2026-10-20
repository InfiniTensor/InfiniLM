#ifndef INFINI_OPS_BASE_ADDMV_H_
#define INFINI_OPS_BASE_ADDMV_H_

#include "operator.h"

namespace infini::ops {

class Addmv : public Operator<Addmv> {
 public:
  Addmv(const Tensor input, const Tensor mat, const Tensor vec,
        const double beta, const double alpha, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        mat_shape_{mat.shape()},
        mat_strides_{mat.strides()},
        mat_type_{mat.dtype()},
        vec_shape_{vec.shape()},
        vec_strides_{vec.strides()},
        vec_type_{vec.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        beta_{beta},
        alpha_{alpha},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input, const Tensor mat,
                          const Tensor vec, const double beta,
                          const double alpha, Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape mat_shape_;

  Tensor::Strides mat_strides_;

  DataType mat_type_;

  Tensor::Shape vec_shape_;

  Tensor::Strides vec_strides_;

  DataType vec_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  double beta_{};

  double alpha_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
