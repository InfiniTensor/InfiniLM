# Core Types

This page summarizes the core types used by InfiniOps operators. InfiniOps
reuses the public InfiniRT runtime types where possible.

## Tensor

`infini::ops::Tensor` is an alias for `infini::rt::TensorView`.

```cpp
#include "tensor.h"

std::vector<float> values(16);
infini::ops::Tensor tensor{
    values.data(),
    std::vector<std::size_t>{4, 4},
    infini::ops::Device{infini::ops::Device::Type::kCpu, 0}};
```

`Tensor` is a non-owning view. It stores the data pointer, shape, data type,
device, and strides, but it does not own the memory it references.

## Device

`infini::ops::Device` is an alias for `infini::rt::Device`.

Known device types include:

- `Device::Type::kCpu`
- `Device::Type::kNvidia`
- `Device::Type::kIluvatar`
- `Device::Type::kMetax`
- `Device::Type::kMoore`
- `Device::Type::kHygon`
- `Device::Type::kCambricon`
- `Device::Type::kAscend`

Some source code also carries experimental or future device enum values. Treat
documented build options in [Backends](../backends.md) as the supported user
surface.

## DataType

`infini::ops::DataType` is imported from InfiniRT. Common values include:

- `DataType::kInt8`, `DataType::kInt16`, `DataType::kInt32`, `DataType::kInt64`
- `DataType::kUInt8`, `DataType::kUInt16`, `DataType::kUInt32`, `DataType::kUInt64`
- `DataType::kFloat16`, `DataType::kBFloat16`, `DataType::kFloat32`, `DataType::kFloat64`
- `DataType::kBool` (distinct from `kUInt8`, with one byte per element)

InfiniOps also exposes type-list helpers such as `FloatTypes`,
`ReducedFloatTypes`, `IntTypes`, `UIntTypes`, and `AllTypes` for template
dispatch.

## Handle

`Handle` carries optional runtime resources for a call:

- backend stream
- workspace pointer
- workspace size

Operators that need temporary memory read the workspace from the handle.
Callers that do not need custom stream or workspace handling can use the
shorter `Op::Call(...)` form without an explicit handle.

## Config

`Config` carries operator configuration that is independent of tensor geometry.
The most common field is `implementation_index`, which selects one of the
active implementations for an operator and device.

```cpp
infini::ops::Config config;
config.set_implementation_index(1);
```

Use the default config unless a specific backend implementation needs to be
selected deliberately.
