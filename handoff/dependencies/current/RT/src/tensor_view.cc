#include "tensor_view.h"

#include <algorithm>
#include <cassert>

#include "dispatcher.h"

namespace infini::rt {

TensorView::TensorView(void* data, std::initializer_list<Size> shape,
                       const DataType& dtype, const Device& device,
                       std::initializer_list<Stride> strides)
    : data_{data},
      shape_strides_storage_{shape, strides},
      dtype_{dtype},
      device_{device} {}

TensorView TensorView::operator[](const Index& index) const {
  const ShapeView shape_view = shape();
  const StridesView strides_view = strides();

  return {reinterpret_cast<decltype(data_)>(
              reinterpret_cast<decltype(index)>(data_) +
              GetEffectiveIndex(index, shape_view[0]) * strides_view[0] *
                  element_size()),
          ShapeView{shape_view.data() + 1, shape_view.size() - 1}, dtype_,
          device_,
          StridesView{strides_view.data() + 1, strides_view.size() - 1}};
}

void*& TensorView::data() { return data_; }

const void* TensorView::data() const { return data_; }

const DataType& TensorView::dtype() const { return dtype_; }

const Device& TensorView::device() const { return device_; }

TensorView::Shape TensorView::shape() && {
  const ShapeView view = shape_strides_storage_.shape();

  return Shape{view.begin(), view.end()};
}

TensorView::Shape TensorView::shape() const&& {
  const ShapeView view = shape_strides_storage_.shape();

  return Shape{view.begin(), view.end()};
}

TensorView::Strides TensorView::strides() && {
  const StridesView view = shape_strides_storage_.strides();

  return Strides{view.begin(), view.end()};
}

TensorView::Strides TensorView::strides() const&& {
  const StridesView view = shape_strides_storage_.strides();

  return Strides{view.begin(), view.end()};
}

TensorView::Size TensorView::element_size() const {
  return kDataTypeToSize.at(dtype_);
}

TensorView TensorView::T() const {
  const ShapeView shape_view = shape();
  const StridesView strides_view = strides();

  return {data_,
          {shape_view[1], shape_view[0]},
          dtype_,
          device_,
          {strides_view[1], strides_view[0]}};
}

std::string TensorView::ToString() const {
  return "tensor(" + ToStringHelper() +
         ", dtype=" + std::string(kDataTypeToDesc.at(dtype_)) + ", device='" +
         device_.ToString() + "')";
}

bool TensorView::HasBroadcastDim() const {
  const ShapeView shape_view = shape();
  const StridesView strides_view = strides();

  return std::any_of(shape_view.begin(), shape_view.end(),
                     [&, i = 0](const auto&) mutable {
                       return shape_view[i] != 1 && strides_view[i++] == 0;
                     });
}

bool TensorView::IsContiguous() const {
  if (ndim() == 0) {
    return true;
  }

  if (!IsMergeable(0, ndim() - 1)) {
    return false;
  }

  return stride(ndim() - 1) == 1;
}

const DataType TensorView::DefaultDataType() { return DataType::kFloat32; }

Device TensorView::DefaultDevice() { return Device{Device::Type::kCpu}; }

std::string TensorView::ToStringHelper() const {
  if (ndim() == 0) {
    return DispatchFunc<Device::Type::kCpu,
                        ConcatType<FloatTypes, AllIntTypes>>(
        dtype_,
        [&](auto tag) {
          using T = typename decltype(tag)::type;
          return std::to_string(*static_cast<T*>(data_));
        },
        "TensorView::ToStringHelper()");
  }

  std::string result{"["};

  for (auto i{Index{0}}; i < shape()[0]; ++i) {
    result += operator[](i).ToStringHelper() + ", ";
  }

  result.pop_back();
  result.back() = ']';

  return result;
}

bool TensorView::IsMergeable(TensorView::Size dim_start,
                             TensorView::Size dim_end) const {
  if (dim_start == dim_end) {
    return true;
  }

  for (TensorView::Size i = dim_start; i < dim_end; ++i) {
    if (size(i) == 1 && stride(i) == 0) {
      return false;
    }
    if (stride(i) != size(i + 1) * stride(i + 1)) {
      return false;
    }
  }

  return true;
}

}  // namespace infini::rt
