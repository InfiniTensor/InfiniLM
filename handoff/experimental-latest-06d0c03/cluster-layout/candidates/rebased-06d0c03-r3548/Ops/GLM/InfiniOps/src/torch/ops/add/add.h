#ifndef INFINI_OPS_TORCH_ADD_H_
#define INFINI_OPS_TORCH_ADD_H_

#include "base/add.h"

namespace infini::ops {

template <Device::Type kDev>
class Operator<Add, kDev, 1> : public Add {
 public:
  Operator(const Tensor input, const Tensor other, const double alpha,
           Tensor out);

  Operator(const Tensor input, const Tensor other, Tensor out);

  void operator()(const Tensor input, const Tensor other, const double alpha,
                  Tensor out) const override;

 private:
  int device_index_{0};
};

}  // namespace infini::ops

#endif
