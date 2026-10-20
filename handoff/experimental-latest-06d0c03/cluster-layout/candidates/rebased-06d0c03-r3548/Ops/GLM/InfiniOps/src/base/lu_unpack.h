#ifndef INFINI_OPS_BASE_LU_UNPACK_H_
#define INFINI_OPS_BASE_LU_UNPACK_H_

#include "operator.h"

namespace infini::ops {

class LuUnpack : public Operator<LuUnpack> {
 public:
  LuUnpack(const Tensor LU_data, const Tensor LU_pivots, const bool unpack_data,
           const bool unpack_pivots, Tensor P, Tensor L, Tensor U)
      : LU_data_shape_{LU_data.shape()},
        LU_data_strides_{LU_data.strides()},
        LU_data_type_{LU_data.dtype()},
        LU_pivots_shape_{LU_pivots.shape()},
        LU_pivots_strides_{LU_pivots.strides()},
        LU_pivots_type_{LU_pivots.dtype()},
        P_shape_{P.shape()},
        P_strides_{P.strides()},
        P_type_{P.dtype()},
        L_shape_{L.shape()},
        L_strides_{L.strides()},
        L_type_{L.dtype()},
        U_shape_{U.shape()},
        U_strides_{U.strides()},
        U_type_{U.dtype()},
        unpack_data_{unpack_data},
        unpack_pivots_{unpack_pivots},
        device_index_{P.device().index()} {}

  virtual void operator()(const Tensor LU_data, const Tensor LU_pivots,
                          const bool unpack_data, const bool unpack_pivots,
                          Tensor P, Tensor L, Tensor U) const = 0;

 protected:
  Tensor::Shape LU_data_shape_;

  Tensor::Strides LU_data_strides_;

  DataType LU_data_type_;

  Tensor::Shape LU_pivots_shape_;

  Tensor::Strides LU_pivots_strides_;

  DataType LU_pivots_type_;

  Tensor::Shape P_shape_;

  Tensor::Strides P_strides_;

  DataType P_type_;

  Tensor::Shape L_shape_;

  Tensor::Strides L_strides_;

  DataType L_type_;

  Tensor::Shape U_shape_;

  Tensor::Strides U_strides_;

  DataType U_type_;

  bool unpack_data_{};

  bool unpack_pivots_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
