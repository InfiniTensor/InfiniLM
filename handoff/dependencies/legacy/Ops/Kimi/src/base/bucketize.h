#ifndef INFINI_OPS_BASE_BUCKETIZE_H_
#define INFINI_OPS_BASE_BUCKETIZE_H_

#include "operator.h"

namespace infini::ops {

class Bucketize : public Operator<Bucketize> {
 public:
  Bucketize(const Tensor input, const Tensor boundaries, const bool out_int32,
            const bool right, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        boundaries_shape_{boundaries.shape()},
        boundaries_strides_{boundaries.strides()},
        boundaries_type_{boundaries.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        out_int32_{out_int32},
        right_{right},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input, const Tensor boundaries,
                          const bool out_int32, const bool right,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape boundaries_shape_;

  Tensor::Strides boundaries_strides_;

  DataType boundaries_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  bool out_int32_{};

  bool right_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
