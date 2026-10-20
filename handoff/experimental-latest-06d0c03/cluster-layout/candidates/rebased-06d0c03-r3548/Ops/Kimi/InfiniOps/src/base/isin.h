#ifndef INFINI_OPS_BASE_ISIN_H_
#define INFINI_OPS_BASE_ISIN_H_

#include "operator.h"

namespace infini::ops {

class Isin : public Operator<Isin> {
 public:
  Isin(const Tensor elements, const Tensor test_elements,
       const bool assume_unique, const bool invert, Tensor out)
      : elements_shape_{elements.shape()},
        elements_strides_{elements.strides()},
        elements_type_{elements.dtype()},
        test_elements_shape_{test_elements.shape()},
        test_elements_strides_{test_elements.strides()},
        test_elements_type_{test_elements.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        assume_unique_{assume_unique},
        invert_{invert},
        device_index_{out.device().index()} {}

  Isin(const Tensor elements, const double test_element,
       const bool assume_unique, const bool invert, Tensor out)
      : elements_shape_{elements.shape()},
        elements_strides_{elements.strides()},
        elements_type_{elements.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        assume_unique_{assume_unique},
        invert_{invert},
        test_element_{test_element},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor elements, const Tensor test_elements,
                          const bool assume_unique, const bool invert,
                          Tensor out) const = 0;

  virtual void operator()(const Tensor elements, const double test_element,
                          const bool assume_unique, const bool invert,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape elements_shape_;

  Tensor::Strides elements_strides_;

  DataType elements_type_;

  Tensor::Shape test_elements_shape_;

  Tensor::Strides test_elements_strides_;

  DataType test_elements_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  bool assume_unique_{};

  bool invert_{};

  double test_element_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
