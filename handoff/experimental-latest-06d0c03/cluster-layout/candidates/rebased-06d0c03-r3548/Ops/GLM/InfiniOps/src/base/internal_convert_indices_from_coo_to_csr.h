#ifndef INFINI_OPS_BASE_INTERNAL_CONVERT_INDICES_FROM_COO_TO_CSR_H_
#define INFINI_OPS_BASE_INTERNAL_CONVERT_INDICES_FROM_COO_TO_CSR_H_

#include "operator.h"

namespace infini::ops::internal {

class ConvertIndicesFromCooToCsr : public Operator<ConvertIndicesFromCooToCsr> {
 public:
  ConvertIndicesFromCooToCsr(const Tensor input, const int64_t size,
                             const bool out_int32, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        size_{size},
        out_int32_{out_int32},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input, const int64_t size,
                          const bool out_int32, Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  int64_t size_{};

  bool out_int32_{};

  int device_index_{0};
};

}  // namespace infini::ops::internal

#endif
