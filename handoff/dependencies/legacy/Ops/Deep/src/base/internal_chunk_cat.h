#ifndef INFINI_OPS_BASE_INTERNAL_CHUNK_CAT_H_
#define INFINI_OPS_BASE_INTERNAL_CHUNK_CAT_H_

#include <vector>

#include "operator.h"

namespace infini::ops::internal {

class ChunkCat : public Operator<ChunkCat> {
 public:
  ChunkCat(const std::vector<Tensor> tensors, const int64_t dim,
           const int64_t num_chunks, Tensor out)
      : out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        tensors_{tensors},
        dim_{dim},
        num_chunks_{num_chunks},
        device_index_{out.device().index()} {}

  virtual void operator()(const std::vector<Tensor> tensors, const int64_t dim,
                          const int64_t num_chunks, Tensor out) const = 0;

 protected:
  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  std::vector<Tensor> tensors_{};

  int64_t dim_{};

  int64_t num_chunks_{};

  int device_index_{0};
};

}  // namespace infini::ops::internal

#endif
