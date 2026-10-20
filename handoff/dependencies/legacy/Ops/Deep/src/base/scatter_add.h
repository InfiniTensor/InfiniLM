#ifndef INFINI_OPS_BASE_SCATTER_ADD_H_
#define INFINI_OPS_BASE_SCATTER_ADD_H_

#include "operator.h"

namespace infini::ops {

class ScatterAdd : public Operator<ScatterAdd> {
 public:
  ScatterAdd(const Tensor input, const Tensor index, const Tensor src,
             const int64_t dim, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        index_shape_{index.shape()},
        index_strides_{index.strides()},
        index_type_{index.dtype()},
        src_shape_{src.shape()},
        src_strides_{src.strides()},
        src_type_{src.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        dim_{dim},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input, const Tensor index,
                          const Tensor src, const int64_t dim,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape index_shape_;

  Tensor::Strides index_strides_;

  DataType index_type_;

  Tensor::Shape src_shape_;

  Tensor::Strides src_strides_;

  DataType src_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  int64_t dim_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
