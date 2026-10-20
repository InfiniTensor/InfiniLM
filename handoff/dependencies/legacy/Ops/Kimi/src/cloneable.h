#ifndef INFINI_OPS_CLONEABLE_H_
#define INFINI_OPS_CLONEABLE_H_

#include <memory>
#include <utility>

namespace infini::ops {

template <typename Base, typename Derived>
class Cloneable : public Base {
 public:
  using Pointer = decltype(std::declval<const Base&>().Clone());

  Pointer Clone() const override {
    return std::make_unique<Derived>(static_cast<const Derived&>(*this));
  }
};

}  // namespace infini::ops

#endif
