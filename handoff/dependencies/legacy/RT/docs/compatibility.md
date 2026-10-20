# Compatibility

This page defines the intended documentation and compatibility boundary for
InfiniRT consumers.

## Stable User Entry

Use:

```cpp
#include <infini/rt.h>
```

The stable user-facing surface is:

- `infini::rt::Device`
- `infini::rt::DataType`
- `infini::rt::TensorView`
- `infini::rt::set_runtime_device_type`
- `infini::rt::runtime_device_type`
- `infini::rt::runtime` types, dispatching functions, and constants documented in
  [Runtime API](api/runtime.md)

## Implementation-Facing Headers

Installed headers under `infini/rt/detail/*` are generated to support the public
entry header. They are not intended as direct user includes.

Backend wrapper headers such as `infini/rt/cpu/runtime_.h` expose
backend-specific `runtime::Runtime<Device::Type::...>` specializations. They are
useful for backend tests and advanced integrations, but ordinary consumers
should prefer `<infini/rt.h>`.

## Generated Headers

Generated public headers depend on the backend options used during CMake
configuration. A consumer should compile against the installed prefix produced
by the same configured build.

## ABI Notes

InfiniRT currently exposes a C++ API. Consumers should treat the installed
headers and `libinfinirt.so` as a matching pair from the same build or release.

`TensorView::Shape` and `TensorView::Strides` are concrete vector-like C++
aliases using inline capacity 8. `TensorView` stores ranks 0 through 8 inline
and uses owned heap fallback at rank 9 and above. This representation changes
`TensorView` layout and is an API/ABI compatibility break from the previous
`std::vector` aliases. Consumers must rebuild after this alias or layout change
and must not mix headers and libraries from different builds.

Existing construction from `std::vector` remains supported, but code that
requires the exact `std::vector` alias must adapt. On lvalues, `shape()` and
`strides()` now return typed borrowed contiguous views by value; callers that
need ownership should explicitly materialize `TensorView::Shape` or
`TensorView::Strides`. The owning aliases support the common
`Strides(count, value)` construction used by downstream metadata code.

