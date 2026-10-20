#include "tensor.h"

#include <algorithm>
#include <cassert>
#include <numeric>

#include "dispatcher.h"

namespace infini::ops {

static Tensor::Index GetEffectiveIndex(Tensor::Index index, Tensor::Size size) {
  return index < 0 ? index + size : index;
}

Tensor::Tensor(void* data, std::initializer_list<Size> shape,
               const DataType& dtype, const Device& device,
               std::initializer_list<Stride> strides)
    : Tensor{data, decltype(shape_){shape}, dtype, device,
             decltype(strides_){strides}} {}

Tensor Tensor::operator[](const Index& index) const {
  return {
      reinterpret_cast<decltype(data_)>(
          reinterpret_cast<decltype(index)>(data_) +
          GetEffectiveIndex(index, shape_[0]) * strides_[0] * element_size()),
      Shape{shape_.cbegin() + 1, shape_.cend()}, dtype_, device_,
      Strides{strides_.cbegin() + 1, strides_.cend()}};
}

void*& Tensor::data() { return data_; }

const void* Tensor::data() const { return data_; }

const Tensor::Shape& Tensor::shape() const { return shape_; }

const DataType& Tensor::dtype() const { return dtype_; }

const Device& Tensor::device() const { return device_; }

const Tensor::Strides& Tensor::strides() const { return strides_; }

Tensor::Size Tensor::size(const Index& index) const {
  return shape_[GetEffectiveIndex(index, shape_.size())];
}

Tensor::Stride Tensor::stride(const Index& index) const {
  return strides_[GetEffectiveIndex(index, strides_.size())];
}

Tensor::Size Tensor::ndim() const { return shape_.size(); }

Tensor::Size Tensor::element_size() const { return kDataTypeToSize.at(dtype_); }

Tensor::Size Tensor::numel() const {
  return std::accumulate(shape_.begin(), shape_.end(),
                         static_cast<Tensor::Size>(1),
                         [](Tensor::Size a, Tensor::Size b) { return a * b; });
}

Tensor Tensor::T() const {
  return {data_,
          {shape_[1], shape_[0]},
          dtype_,
          device_,
          {strides_[1], strides_[0]}};
}

std::string Tensor::ToString() const {
  return "tensor(" + ToStringHelper() +
         ", dtype=" + std::string(kDataTypeToDesc.at(dtype_)) + ", device='" +
         device_.ToString() + "')";
}

bool Tensor::HasBroadcastDim() const {
  return std::any_of(shape_.begin(), shape_.end(),
                     [&, i = 0](const auto&) mutable {
                       return shape_[i] != 1 && strides_[i++] == 0;
                     });
}

bool Tensor::IsContiguous() const {
  if (ndim() == 0) {
    return true;
  }

  if (!IsMergeable(0, ndim() - 1)) {
    return false;
  }

  return stride(ndim() - 1) == 1;
}

const DataType Tensor::DefaultDataType() { return DataType::kFloat32; }

Device Tensor::DefaultDevice() { return Device{Device::Type::kCpu}; }

Tensor::Strides Tensor::DefaultStrides(const Shape& shape) {
  if (shape.empty()) {
    return {};
  }

  Strides strides(shape.size());

  strides.back() = 1;

  for (auto i{shape.size() - 2}; i != -1; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }

  return strides;
}

std::string Tensor::ToStringHelper() const {
  if (ndim() == 0) {
    return DispatchFunc<Device::Type::kCpu,
                        ConcatType<FloatTypes, AllIntTypes>>(
        dtype_,
        [&](auto tag) {
          using T = typename decltype(tag)::type;
          return std::to_string(*static_cast<T*>(data_));
        },
        "Tensor::ToStringHelper()");
  }

  std::string result{"["};

  for (auto i{Index{0}}; i < shape_[0]; ++i) {
    result += operator[](i).ToStringHelper() + ", ";
  }

  result.pop_back();
  result.back() = ']';

  return result;
}

bool Tensor::IsMergeable(Tensor::Size dim_start, Tensor::Size dim_end) const {
  if (dim_start == dim_end) {
    return true;
  }

  for (Tensor::Size i = dim_start; i < dim_end; ++i) {
    if (size(i) == 1 && stride(i) == 0) {
      return false;
    }
    if (stride(i) != size(i + 1) * stride(i + 1)) {
      return false;
    }
  }

  return true;
}

}  // namespace infini::ops
