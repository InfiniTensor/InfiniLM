#ifndef INFINI_OPS_TORCH_TENSOR__H_
#define INFINI_OPS_TORCH_TENSOR__H_

#include <torch/torch.h>
#include <torch/version.h>

#include <optional>
#include <vector>

#include "tensor.h"
#include "torch/device_.h"

namespace infini::ops {

namespace detail {

// Introduces a dependent type alias for `c10::ScalarType`, allowing
// `if constexpr` to discard branches that reference enum values absent
// in older PyTorch versions.  Without this indirection the enum member
// names (e.g. `UInt16`) are non-dependent and looked up at definition
// time, causing a hard error on PyTorch < 2.4.
template <int>
struct DependentScalarType {
  using type = c10::ScalarType;
};

constexpr int kTorchVersion = TORCH_VERSION_MAJOR * 100 + TORCH_VERSION_MINOR;

// Unsigned integer scalar types are only available in PyTorch >= 2.4.
template <int kVersion = kTorchVersion>
inline at::ScalarType ToAtenUnsignedDataType(DataType dtype) {
  if constexpr (kVersion >= 204) {
    using ST = typename DependentScalarType<kVersion>::type;
    switch (dtype) {
      case DataType::kUInt16:
        return ST::UInt16;
      case DataType::kUInt32:
        return ST::UInt32;
      case DataType::kUInt64:
        return ST::UInt64;
      default:
        assert(false && "not an unsigned integer dtype");
        return at::kFloat;
    }
  } else {
    (void)dtype;
    assert(false && "unsigned integer types require PyTorch 2.4 or later");
    return at::kFloat;
  }
}

}  // namespace detail

inline at::ScalarType ToAtenDataType(DataType dtype) {
  switch (dtype) {
    case DataType::kInt8:
      return at::kChar;
    case DataType::kInt16:
      return at::kShort;
    case DataType::kInt32:
      return at::kInt;
    case DataType::kInt64:
      return at::kLong;
    case DataType::kUInt8:
      return at::kByte;
    case DataType::kBool:
      return at::kBool;
    case DataType::kUInt16:
    case DataType::kUInt32:
    case DataType::kUInt64:
      return detail::ToAtenUnsignedDataType(dtype);
    case DataType::kFloat16:
      return at::kHalf;
    case DataType::kBFloat16:
      return at::kBFloat16;
    case DataType::kFloat32:
      return at::kFloat;
    case DataType::kFloat64:
      return at::kDouble;
    default:
      assert(false && "unsupported dtype for ATen conversion");
      return at::kFloat;
  }
}

// Build an ATen tensor from explicit metadata. Use this instead of reading
// shape/strides from the `Tensor` parameter, which may have been moved-from
// by the `Call()` dispatch path (see `operator.h`).
template <Device::Type kDev, typename ShapeRange, typename StridesRange>
inline at::Tensor ToAtenTensor(
    void* data, const ShapeRange& shape, const StridesRange& strides,
    DataType dtype, int device_index = 0,
    std::optional<at::ScalarType> dtype_override = std::nullopt) {
  std::vector<int64_t> at_shape(shape.begin(), shape.end());
  std::vector<int64_t> at_strides(strides.begin(), strides.end());

  auto options =
      at::TensorOptions().dtype(dtype_override.value_or(ToAtenDataType(dtype)));

  if constexpr (kDev != Device::Type::kCpu) {
    std::string device_str =
        std::string(detail::TorchDeviceName<kDev>::kValue) + ":" +
        std::to_string(device_index);
    options = options.device(device_str);
  }

  for (auto dim : shape) {
    if (dim == 0) return at::empty_strided(at_shape, at_strides, options);
  }

  return at::from_blob(data, at_shape, at_strides, options);
}

}  // namespace infini::ops

#endif
