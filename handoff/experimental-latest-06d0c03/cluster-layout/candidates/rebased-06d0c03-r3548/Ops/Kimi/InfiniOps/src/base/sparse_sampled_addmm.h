#ifndef INFINI_OPS_BASE_SPARSE_SAMPLED_ADDMM_H_
#define INFINI_OPS_BASE_SPARSE_SAMPLED_ADDMM_H_

#include "operator.h"

namespace infini::ops {

class SparseSampledAddmm : public Operator<SparseSampledAddmm> {
 public:
  SparseSampledAddmm(const Tensor input, const Tensor mat1, const Tensor mat2,
                     const double beta, const double alpha, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        mat1_shape_{mat1.shape()},
        mat1_strides_{mat1.strides()},
        mat1_type_{mat1.dtype()},
        mat2_shape_{mat2.shape()},
        mat2_strides_{mat2.strides()},
        mat2_type_{mat2.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        beta_{beta},
        alpha_{alpha},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input, const Tensor mat1,
                          const Tensor mat2, const double beta,
                          const double alpha, Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape mat1_shape_;

  Tensor::Strides mat1_strides_;

  DataType mat1_type_;

  Tensor::Shape mat2_shape_;

  Tensor::Strides mat2_strides_;

  DataType mat2_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  double beta_{};

  double alpha_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
