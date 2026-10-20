#ifndef INFINI_OPS_BASE_SVD_H_
#define INFINI_OPS_BASE_SVD_H_

#include "operator.h"

namespace infini::ops {

class Svd : public Operator<Svd> {
 public:
  Svd(const Tensor input, const bool some, const bool compute_uv, Tensor U,
      Tensor S, Tensor V)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        U_shape_{U.shape()},
        U_strides_{U.strides()},
        U_type_{U.dtype()},
        S_shape_{S.shape()},
        S_strides_{S.strides()},
        S_type_{S.dtype()},
        V_shape_{V.shape()},
        V_strides_{V.strides()},
        V_type_{V.dtype()},
        some_{some},
        compute_uv_{compute_uv},
        device_index_{U.device().index()} {}

  virtual void operator()(const Tensor input, const bool some,
                          const bool compute_uv, Tensor U, Tensor S,
                          Tensor V) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape U_shape_;

  Tensor::Strides U_strides_;

  DataType U_type_;

  Tensor::Shape S_shape_;

  Tensor::Strides S_strides_;

  DataType S_type_;

  Tensor::Shape V_shape_;

  Tensor::Strides V_strides_;

  DataType V_type_;

  bool some_{};

  bool compute_uv_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
