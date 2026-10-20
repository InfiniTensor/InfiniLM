#ifndef INFINI_OPS_BASE_VSTACK_H_
#define INFINI_OPS_BASE_VSTACK_H_

#include <vector>

#include "operator.h"

namespace infini::ops {

class Vstack : public Operator<Vstack> {
 public:
  Vstack(const std::vector<Tensor> tensors, Tensor out)
      : out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        tensors_{tensors},
        device_index_{out.device().index()} {}

  virtual void operator()(const std::vector<Tensor> tensors,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  std::vector<Tensor> tensors_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
