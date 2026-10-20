#ifndef INFINI_OPS_BASE_INDEX_H_
#define INFINI_OPS_BASE_INDEX_H_

#include <optional>
#include <vector>

#include "operator.h"

namespace infini::ops {

class Index : public Operator<Index> {
 public:
  Index(const Tensor input, const std::vector<std::optional<Tensor>> indices,
        Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        indices_{indices},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input,
                          const std::vector<std::optional<Tensor>> indices,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  std::vector<std::optional<Tensor>> indices_{};

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  int device_index_{0};
};

}  // namespace infini::ops

#endif
