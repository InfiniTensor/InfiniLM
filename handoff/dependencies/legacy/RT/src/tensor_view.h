#ifndef INFINI_RT_TENSOR_VIEW_H_
#define INFINI_RT_TENSOR_VIEW_H_

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <type_traits>
#include <utility>

#include "common/shape_strides_storage.h"
#include "data_type.h"
#include "device.h"
#include "hash.h"

namespace infini::rt {

namespace tensor_view_detail {

inline constexpr std::size_t kInlineMetadataCapacity = 8;

template <typename T, typename = void>
struct IsTensorLike : std::false_type {};

template <typename T>
struct IsTensorLike<T,
                    std::void_t<decltype(std::declval<const T&>().data()),
                                decltype(std::declval<const T&>().shape()),
                                decltype(std::declval<const T&>().dtype()),
                                decltype(std::declval<const T&>().device()),
                                decltype(std::declval<const T&>().strides())>>
    : std::true_type {};

}  // namespace tensor_view_detail

class TensorView {
 public:
  using Size = std::size_t;

  using Stride = std::ptrdiff_t;

  using Index = Stride;

 private:
  using ShapeStridesStorage =
      detail::ShapeStridesStorage<Size, Stride,
                                  tensor_view_detail::kInlineMetadataCapacity>;

 public:
  using Shape = typename ShapeStridesStorage::Shape;

  using Strides = typename ShapeStridesStorage::Strides;

  using ShapeView = typename ShapeStridesStorage::ShapeView;

  using StridesView = typename ShapeStridesStorage::StridesView;

  template <typename TensorLike,
            typename = std::enable_if_t<
                tensor_view_detail::IsTensorLike<TensorLike>::value>>
  TensorView(const TensorLike& tensor)
      : data_{const_cast<void*>(static_cast<const void*>(tensor.data()))},
        shape_strides_storage_{tensor.shape(), tensor.strides()},
        dtype_{tensor.dtype()},
        device_{tensor.device()} {}

  TensorView(void* data, const Shape& shape)
      : data_{data},
        shape_strides_storage_{shape, detail::DefaultStridesTag{}},
        dtype_{DefaultDataType()},
        device_{DefaultDevice()} {}

  TensorView(void* data, Shape&& shape)
      : data_{data},
        shape_strides_storage_{std::move(shape), detail::DefaultStridesTag{}},
        dtype_{DefaultDataType()},
        device_{DefaultDevice()} {}

  template <typename ShapeLike>
  TensorView(void* data, const ShapeLike& shape)
      : data_{data},
        shape_strides_storage_{shape, detail::DefaultStridesTag{}},
        dtype_{DefaultDataType()},
        device_{DefaultDevice()} {}

  TensorView(void* data, const Shape& shape, const DataType& dtype)
      : data_{data},
        shape_strides_storage_{shape, detail::DefaultStridesTag{}},
        dtype_{dtype},
        device_{DefaultDevice()} {}

  TensorView(void* data, Shape&& shape, const DataType& dtype)
      : data_{data},
        shape_strides_storage_{std::move(shape), detail::DefaultStridesTag{}},
        dtype_{dtype},
        device_{DefaultDevice()} {}

  template <typename ShapeLike>
  TensorView(void* data, const ShapeLike& shape, const DataType& dtype)
      : data_{data},
        shape_strides_storage_{shape, detail::DefaultStridesTag{}},
        dtype_{dtype},
        device_{DefaultDevice()} {}

  TensorView(void* data, const Shape& shape, const Device& device)
      : data_{data},
        shape_strides_storage_{shape, detail::DefaultStridesTag{}},
        dtype_{DefaultDataType()},
        device_{device} {}

  TensorView(void* data, Shape&& shape, const Device& device)
      : data_{data},
        shape_strides_storage_{std::move(shape), detail::DefaultStridesTag{}},
        dtype_{DefaultDataType()},
        device_{device} {}

  template <typename ShapeLike>
  TensorView(void* data, const ShapeLike& shape, const Device& device)
      : data_{data},
        shape_strides_storage_{shape, detail::DefaultStridesTag{}},
        dtype_{DefaultDataType()},
        device_{device} {}

  TensorView(void* data, const Shape& shape, const DataType& dtype,
             const Device& device)
      : data_{data},
        shape_strides_storage_{shape, detail::DefaultStridesTag{}},
        dtype_{dtype},
        device_{device} {}

  TensorView(void* data, Shape&& shape, const DataType& dtype,
             const Device& device)
      : data_{data},
        shape_strides_storage_{std::move(shape), detail::DefaultStridesTag{}},
        dtype_{dtype},
        device_{device} {}

  template <typename ShapeLike>
  TensorView(void* data, const ShapeLike& shape, const DataType& dtype,
             const Device& device)
      : data_{data},
        shape_strides_storage_{shape, detail::DefaultStridesTag{}},
        dtype_{dtype},
        device_{device} {}

  TensorView(void* data, const Shape& shape, const DataType& dtype,
             const Device& device, const Strides& strides)
      : data_{data},
        shape_strides_storage_{shape, strides},
        dtype_{dtype},
        device_{device} {}

  TensorView(void* data, Shape&& shape, const DataType& dtype,
             const Device& device, Strides&& strides)
      : data_{data},
        shape_strides_storage_{std::move(shape), std::move(strides)},
        dtype_{dtype},
        device_{device} {}

  TensorView(void* data, Shape&& shape, const DataType& dtype,
             const Device& device, const Strides& strides)
      : data_{data},
        shape_strides_storage_{std::move(shape), strides},
        dtype_{dtype},
        device_{device} {}

  TensorView(void* data, const Shape& shape, const DataType& dtype,
             const Device& device, Strides&& strides)
      : data_{data},
        shape_strides_storage_{shape, std::move(strides)},
        dtype_{dtype},
        device_{device} {}

  template <typename ShapeLike, typename StridesLike>
  TensorView(void* data, const ShapeLike& shape, const DataType& dtype,
             const Device& device, const StridesLike& strides)
      : data_{data},
        shape_strides_storage_{shape, strides},
        dtype_{dtype},
        device_{device} {}

  TensorView(void* data, std::initializer_list<Size> shape,
             const DataType& dtype, const Device& device,
             std::initializer_list<Stride> strides);

  TensorView operator[](const Index& index) const;

  void*& data();

  const void* data() const;

  const DataType& dtype() const;

  const Device& device() const;

  ShapeView shape() const& noexcept { return shape_strides_storage_.shape(); }

  Shape shape() &&;

  Shape shape() const&&;

  StridesView strides() const& noexcept {
    return shape_strides_storage_.strides();
  }

  Strides strides() &&;

  Strides strides() const&&;

  Size size(const Index& index) const noexcept {
    const Size rank = shape_strides_storage_.shape_size();

    return shape_strides_storage_
        .shape_data()[static_cast<Size>(GetEffectiveIndex(index, rank))];
  }

  Stride stride(const Index& index) const noexcept {
    const Size rank = shape_strides_storage_.strides_size();

    return shape_strides_storage_
        .strides_data()[static_cast<Size>(GetEffectiveIndex(index, rank))];
  }

  Size ndim() const noexcept { return shape_strides_storage_.shape_size(); }

  Size element_size() const;

  Size numel() const noexcept {
    const Size rank = shape_strides_storage_.shape_size();
    const Size* const shape_data = shape_strides_storage_.shape_data();
    Size result = 1;

    for (Size axis = 0; axis < rank; ++axis) {
      result *= shape_data[axis];
    }

    return result;
  }

  TensorView T() const;

  std::string ToString() const;

  bool HasBroadcastDim() const;

  bool IsContiguous() const;

 private:
  static constexpr Index GetEffectiveIndex(Index index, Size size) noexcept {
    return index < 0 ? index + static_cast<Index>(size) : index;
  }

  static const DataType DefaultDataType();

  static Device DefaultDevice();

  std::string ToStringHelper() const;

  bool IsMergeable(Size dim_start, Size dim_end) const;

  void* data_{nullptr};

  ShapeStridesStorage shape_strides_storage_;

  const DataType dtype_;

  Device device_;
};

}  // namespace infini::rt

template <>
struct std::hash<infini::rt::TensorView> {
  std::size_t operator()(const infini::rt::TensorView& tensor) const {
    std::size_t seed{0};

    for (const auto& size : tensor.shape()) {
      HashCombine(seed, size);
    }

    HashCombine(seed, tensor.dtype());

    HashCombine(seed, tensor.device());

    for (const auto& stride : tensor.strides()) {
      HashCombine(seed, stride);
    }

    return seed;
  }
};

template <>
struct std::equal_to<infini::rt::TensorView> {
  bool operator()(const infini::rt::TensorView& a,
                  const infini::rt::TensorView& b) const {
    return a.dtype() == b.dtype() && a.device() == b.device() &&
           a.shape() == b.shape() && a.strides() == b.strides();
  }
};

#endif
