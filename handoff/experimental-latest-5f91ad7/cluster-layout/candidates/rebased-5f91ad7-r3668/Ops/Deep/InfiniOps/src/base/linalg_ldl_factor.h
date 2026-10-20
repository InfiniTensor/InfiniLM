#ifndef INFINI_OPS_BASE_LINALG_LDL_FACTOR_H_
#define INFINI_OPS_BASE_LINALG_LDL_FACTOR_H_

#include "operator.h"

namespace infini::ops::linalg {

class LdlFactor : public Operator<LdlFactor> {
 public:
  LdlFactor(const Tensor input, const bool hermitian, Tensor LD, Tensor pivots)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        LD_shape_{LD.shape()},
        LD_strides_{LD.strides()},
        LD_type_{LD.dtype()},
        pivots_shape_{pivots.shape()},
        pivots_strides_{pivots.strides()},
        pivots_type_{pivots.dtype()},
        hermitian_{hermitian},
        device_index_{LD.device().index()} {}

  virtual void operator()(const Tensor input, const bool hermitian, Tensor LD,
                          Tensor pivots) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape LD_shape_;

  Tensor::Strides LD_strides_;

  DataType LD_type_;

  Tensor::Shape pivots_shape_;

  Tensor::Strides pivots_strides_;

  DataType pivots_type_;

  bool hermitian_{};

  int device_index_{0};
};

}  // namespace infini::ops::linalg

#endif
