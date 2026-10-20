#ifndef INFINI_OPS_NINETOOTHED_TENSOR_H_
#define INFINI_OPS_NINETOOTHED_TENSOR_H_

#include <cstdint>
#include <type_traits>

#include "data_type.h"
#include "tensor.h"

namespace infini::ops::ninetoothed {

inline int DataTypeIndex(DataType dtype) {
  switch (dtype) {
    case DataType::kFloat16:
      return 8;
    case DataType::kBFloat16:
      return 9;
    case DataType::kFloat32:
      return 10;
    default:
      return -1;
  }
}

class Tensor {
 public:
  explicit Tensor(const ::infini::ops::Tensor& tensor)
      : Tensor(const_cast<void*>(tensor.data()),
               reinterpret_cast<std::uint64_t*>(
                   const_cast<::infini::ops::Tensor::Size*>(
                       tensor.shape().data())),
               reinterpret_cast<std::int64_t*>(
                   const_cast<::infini::ops::Tensor::Stride*>(
                       tensor.strides().data()))) {
    static_assert(sizeof(::infini::ops::Tensor::Size) == sizeof(std::uint64_t));
    static_assert(sizeof(::infini::ops::Tensor::Stride) ==
                  sizeof(std::int64_t));
    static_assert(std::is_unsigned_v<::infini::ops::Tensor::Size>);
    static_assert(std::is_signed_v<::infini::ops::Tensor::Stride>);
  }

  Tensor(void* data, std::uint64_t* shape, std::int64_t* strides)
      : data_(data), shape_(shape), strides_(strides) {}

  template <typename T>
  Tensor(T& value, std::uint64_t* shape, std::int64_t* strides)
      : Tensor(static_cast<void*>(&value), shape, strides) {}

  template <typename NineToothedTensor>
  operator NineToothedTensor() const {
    return NineToothedTensor{data_, shape_, strides_};
  }

 private:
  void* data_;

  std::uint64_t* shape_;

  std::int64_t* strides_;
};

}  // namespace infini::ops::ninetoothed

#endif
