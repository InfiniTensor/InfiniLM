#ifndef INFINI_OPS_BASE_SET_H_
#define INFINI_OPS_BASE_SET_H_

#include <vector>

#include "operator.h"

namespace infini::ops {

class Set : public Operator<Set> {
 public:
  Set(const Tensor input, const Tensor source, const int64_t storage_offset,
      const std::vector<int64_t> size, const std::vector<int64_t> stride,
      Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        source_shape_{source.shape()},
        source_strides_{source.strides()},
        source_type_{source.dtype()},
        storage_offset_{storage_offset},
        size_{size},
        stride_{stride},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        device_index_{out.device().index()} {}

  Set(const Tensor input, const Tensor source, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        source_shape_{source.shape()},
        source_strides_{source.strides()},
        source_type_{source.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input, const Tensor source,
                          const int64_t storage_offset,
                          const std::vector<int64_t> size,
                          const std::vector<int64_t> stride,
                          Tensor out) const = 0;

  virtual void operator()(const Tensor input, const Tensor source,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape source_shape_;

  Tensor::Strides source_strides_;

  DataType source_type_;

  int64_t storage_offset_{};

  std::vector<int64_t> size_{};

  std::vector<int64_t> stride_{};

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  int device_index_{0};
};

}  // namespace infini::ops

#endif
