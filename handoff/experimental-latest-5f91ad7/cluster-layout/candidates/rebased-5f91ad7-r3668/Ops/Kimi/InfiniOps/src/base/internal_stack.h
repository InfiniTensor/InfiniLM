#ifndef INFINI_OPS_BASE_INTERNAL_STACK_H_
#define INFINI_OPS_BASE_INTERNAL_STACK_H_

#include <vector>

#include "operator.h"

namespace infini::ops::internal {

class Stack : public Operator<Stack> {
 public:
  Stack(const std::vector<Tensor> tensors, const int64_t dim, Tensor out)
      : out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        tensors_{tensors},
        dim_{dim},
        device_index_{out.device().index()} {}

  virtual void operator()(const std::vector<Tensor> tensors, const int64_t dim,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  std::vector<Tensor> tensors_{};

  int64_t dim_{};

  int device_index_{0};
};

}  // namespace infini::ops::internal

#endif
