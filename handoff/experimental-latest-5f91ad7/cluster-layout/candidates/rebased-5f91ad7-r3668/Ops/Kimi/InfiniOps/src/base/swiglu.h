#ifndef INFINI_OPS_BASE_SWIGLU_H_
#define INFINI_OPS_BASE_SWIGLU_H_

#include <optional>

#include "operator.h"

namespace infini::ops {

/// \deprecated Use `SiluAndMul`. This interface will be removed in a future
/// release.
class [[deprecated("Use `SiluAndMul` instead.")]] Swiglu
    : public Operator<Swiglu> {
 public:
  Swiglu(const Tensor input, const Tensor gate, Tensor out)
      : ndim_{out.ndim()},
        output_size_{out.numel()},
        input_type_{input.dtype()},
        gate_type_{gate.dtype()},
        out_type_{out.dtype()},
        input_shape_{input.shape()},
        gate_shape_{gate.shape()},
        out_shape_{out.shape()},
        input_strides_{input.strides()},
        gate_strides_{gate.strides()},
        out_strides_{out.strides()},
        is_input_contiguous_{input.IsContiguous()},
        is_gate_contiguous_{gate.IsContiguous()},
        is_out_contiguous_{out.IsContiguous()} {
    assert(
        input_type_ == gate_type_ && gate_type_ == out_type_ &&
        "operator `Swiglu` requires all input and output tensors to have the "
        "same dtype");
  }

  virtual void operator()(const Tensor input, const Tensor gate,
                          Tensor out) const = 0;

 protected:
  Tensor::Size ndim_{0};

  Tensor::Size output_size_{0};

  const DataType input_type_;

  const DataType gate_type_;

  const DataType out_type_;

  Tensor::Shape input_shape_;

  Tensor::Shape gate_shape_;

  Tensor::Shape out_shape_;

  Tensor::Strides input_strides_;

  Tensor::Strides gate_strides_;

  Tensor::Strides out_strides_;

  bool is_input_contiguous_{false};

  bool is_gate_contiguous_{false};

  bool is_out_contiguous_{false};
};

}  // namespace infini::ops

#endif
