#ifndef INFINI_OPS_BASE_INDEX_ADD_H_
#define INFINI_OPS_BASE_INDEX_ADD_H_

#include "operator.h"

namespace infini::ops {

class IndexAdd : public Operator<IndexAdd> {
 public:
  IndexAdd(const Tensor input, const Tensor index, const Tensor source,
           const int64_t dim, const double alpha, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        index_shape_{index.shape()},
        index_strides_{index.strides()},
        index_type_{index.dtype()},
        source_shape_{source.shape()},
        source_strides_{source.strides()},
        source_type_{source.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        dim_{dim},
        alpha_{alpha},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input, const Tensor index,
                          const Tensor source, const int64_t dim,
                          const double alpha, Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape index_shape_;

  Tensor::Strides index_strides_;

  DataType index_type_;

  Tensor::Shape source_shape_;

  Tensor::Strides source_strides_;

  DataType source_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  int64_t dim_{};

  double alpha_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
