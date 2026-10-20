#ifndef INFINI_OPS_BASE_INTERNAL_CONVERT_INDICES_FROM_CSR_TO_COO_H_
#define INFINI_OPS_BASE_INTERNAL_CONVERT_INDICES_FROM_CSR_TO_COO_H_

#include "operator.h"

namespace infini::ops::internal {

class ConvertIndicesFromCsrToCoo : public Operator<ConvertIndicesFromCsrToCoo> {
 public:
  ConvertIndicesFromCsrToCoo(const Tensor crow_indices,
                             const Tensor col_indices, const bool out_int32,
                             const bool transpose, Tensor out)
      : crow_indices_shape_{crow_indices.shape()},
        crow_indices_strides_{crow_indices.strides()},
        crow_indices_type_{crow_indices.dtype()},
        col_indices_shape_{col_indices.shape()},
        col_indices_strides_{col_indices.strides()},
        col_indices_type_{col_indices.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        out_int32_{out_int32},
        transpose_{transpose},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor crow_indices, const Tensor col_indices,
                          const bool out_int32, const bool transpose,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape crow_indices_shape_;

  Tensor::Strides crow_indices_strides_;

  DataType crow_indices_type_;

  Tensor::Shape col_indices_shape_;

  Tensor::Strides col_indices_strides_;

  DataType col_indices_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  bool out_int32_{};

  bool transpose_{};

  int device_index_{0};
};

}  // namespace infini::ops::internal

#endif
