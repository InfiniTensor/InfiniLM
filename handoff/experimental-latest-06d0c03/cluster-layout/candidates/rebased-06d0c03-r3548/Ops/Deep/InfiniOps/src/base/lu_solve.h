#ifndef INFINI_OPS_BASE_LU_SOLVE_H_
#define INFINI_OPS_BASE_LU_SOLVE_H_

#include "operator.h"

namespace infini::ops {

class LuSolve : public Operator<LuSolve> {
 public:
  LuSolve(const Tensor input, const Tensor LU_data, const Tensor LU_pivots,
          Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        LU_data_shape_{LU_data.shape()},
        LU_data_strides_{LU_data.strides()},
        LU_data_type_{LU_data.dtype()},
        LU_pivots_shape_{LU_pivots.shape()},
        LU_pivots_strides_{LU_pivots.strides()},
        LU_pivots_type_{LU_pivots.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input, const Tensor LU_data,
                          const Tensor LU_pivots, Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape LU_data_shape_;

  Tensor::Strides LU_data_strides_;

  DataType LU_data_type_;

  Tensor::Shape LU_pivots_shape_;

  Tensor::Strides LU_pivots_strides_;

  DataType LU_pivots_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  int device_index_{0};
};

}  // namespace infini::ops

#endif
