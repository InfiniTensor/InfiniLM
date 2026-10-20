#ifndef INFINI_OPS_BASE_NARROW_COPY_H_
#define INFINI_OPS_BASE_NARROW_COPY_H_

#include "operator.h"

namespace infini::ops {

class NarrowCopy : public Operator<NarrowCopy> {
 public:
  NarrowCopy(const Tensor input, const int64_t dim, const int64_t start,
             const int64_t length, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        dim_{dim},
        start_{start},
        length_{length},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input, const int64_t dim,
                          const int64_t start, const int64_t length,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  int64_t dim_{};

  int64_t start_{};

  int64_t length_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
