# Adding ATen-backed operators

InfiniOps can expose PyTorch ATen kernels as an InfiniOps backend when
`WITH_TORCH=ON`. This is intended for broad operator coverage and for validating
the generated wrapper path before a platform-native implementation exists.

The generated PyTorch backend is derived from the ATen schema bundled with the
locally installed PyTorch package. No network fetch is required:
`WITH_TORCH=ON` already requires PyTorch, and PyTorch wheels ship the matching
`torchgen/packaged/ATen/native/native_functions.yaml`. When generation is not a
good fit, a PyTorch backend can also be written by hand under `src/torch/`.

## Choose generated or hand-written

There are two supported ways to use ATen inside InfiniOps:

- Generated ATen backend: add an operator to `scripts/torch_ops.yaml` and let
  `scripts/generate_torch_ops.py` emit the base and PyTorch backend.
- Hand-written ATen backend: write a normal InfiniOps backend specialization
  under `src/torch/ops/<op>/` and call ATen manually.

Prefer generation when the public InfiniOps API can mirror an ATen `.out`
schema. Use a hand-written ATen backend when the operator already has a
hand-written base, needs an InfiniOps-specific API, needs special fallback
logic, or does not correspond to a single ATen `.out` call.

Existing hand-written ATen examples include `src/torch/ops/add/` and
`src/torch/ops/gemm/`.

## Generated backend pieces

The ATen generator is `scripts/generate_torch_ops.py`. For each selected op it
tries to find one or more usable ATen `.out` schemas and writes generated files
under `generated/`:

- `generated/base/<op>.h`: an InfiniOps operator base class when no
  hand-written `src/base/<op>.h` exists.
- `generated/torch/<op>/<op>.h` and `.cc`: the PyTorch backend implementation.
- `generated/torch_ops_metadata.json`: metadata consumed by
  `tests/test_torch_ops.py`.

These generated files are build artifacts and are not committed. CMake
regenerates them at configure time when `WITH_TORCH=ON`.

The PyTorch implementation uses implementation index `8`. Native and vendor
implementations use indices `0` through `7`. This convention applies to
generated ATen wrappers. Hand-written ATen backends may use another explicit
implementation index, but must avoid colliding with the operator's existing
implementations.

## Add a generated ATen operator

1. Make sure the target environment has PyTorch and `torchgen` installed.
   Vendor PyTorch forks are supported as long as they ship the matching
   packaged ATen schema.

2. Add the ATen base name to `scripts/torch_ops.yaml`.

   Use the ATen name without the `_out` suffix. For example:

   ```yaml
   - clamp_min
   ```

   To try an operator without editing the allowlist, pass it through
   `INFINI_OPS_TORCH_OPS` or the generator's `--ops` argument.

3. Run the generator locally or through a CMake configure:

   ```bash
   python3 scripts/generate_torch_ops.py --ops clamp_min
   ```

   Inspect the output under `generated/base/`, `generated/torch/`, and
   `generated/torch_ops_metadata.json`.

4. Build with the PyTorch backend enabled. For focused iteration, use a smoke
   build and explicitly allowlist the operator for both wrapper generation and
   ATen wrapper generation:

   ```bash
   python3 -m pip install --no-build-isolation --no-deps . \
     --config-settings=cmake.define.WITH_CPU=ON \
     --config-settings=cmake.define.WITH_TORCH=ON \
     --config-settings=cmake.define.INFINI_OPS_SMOKE_BUILD=ON \
     --config-settings=cmake.define.INFINI_OPS_OPS=clamp_min \
     --config-settings=cmake.define.INFINI_OPS_TORCH_OPS=clamp_min
   ```

   Platform validation should also enable the platform backend, for example
   `WITH_NVIDIA=ON`, `WITH_ASCEND=ON`, and so on.

5. Run generated coverage:

   ```bash
   python3 -m pytest tests/test_torch_ops.py -q -v --devices cpu <platform>
   ```

   Add or update a focused handwritten test when the operator is expected to
   become part of the normal operator test set. The generated test proves the
   ATen wrapper path is wired; handwritten tests provide operator-specific
   coverage with stable shapes, dtypes, tolerances, and skip rules.

## How schemas become InfiniOps APIs

The generator only uses ATen `.out` schemas. Output tensors become InfiniOps
`Tensor out` parameters. ATen's `self` parameter is exposed as `input` in C++.

Supported ATen scalar and container types are mapped to torch-independent
InfiniOps-facing C++ types. Common examples:

| ATen schema type | InfiniOps-facing type |
| ---------------- | --------------------- |
| `Tensor` | `Tensor` |
| `Tensor[]` | `std::vector<Tensor>` |
| `Scalar` / `float` | `double` |
| `int` / `SymInt` | `int64_t` |
| `bool` | `bool` |
| `str` | `std::string` |
| `ScalarType` | `DataType` |
| `int[]` / `SymInt[]` | `std::vector<int64_t>` |

Optional types with stable InfiniOps representations are exposed as
`std::optional<...>`, such as `std::optional<Tensor>`,
`std::optional<double>`, or `std::optional<std::vector<int64_t>>`.

Optional ATen-only concepts that do not have a stable public InfiniOps
representation, such as `MemoryFormat?`, `Layout?`, `Device?`, and
`Generator?`, stay hidden and are forwarded to ATen as typed empty optionals.

## Existing base classes

If `src/base/<op>.h` already exists, the generated base is not emitted for that
operator. Instead, the generator parses the existing `operator()` overloads and
tries to bind each overload to a usable ATen schema.

This lets hand-written base classes define the public API while still using an
ATen backend.

Binding rules:

- Parameter names and compatible C++ types must match in order.
- ATen parameters can be skipped only when they are optional or have defaults.
- If several ATen overloads match, the one with fewer hidden/defaulted
  parameters is preferred.
- If no existing overload matches, the PyTorch backend for that operator is
  skipped and a warning is printed.

This is useful when a base class intentionally exposes a smaller overload and
the missing ATen parameters can safely use defaults. If the missing parameter is
semantically important, add it to the base overload instead of hiding it.

## Naming rules

ATen overload suffixes such as `Tensor_out`, `out_x`, or `grad_input` are not
part of the public InfiniOps class name. Distinct ATen overloads for the same
public operator become overloaded constructors and `operator()` methods on one
class.

Namespace-style prefixes are represented as C++ namespaces:

- `special_erfinv` -> `infini::ops::special::Erfinv`
- `linalg_det` -> `infini::ops::linalg::Det`
- `fft_fft` -> `infini::ops::fft::Fft`
- internal ATen names with a leading underscore are placed under
  `infini::ops::internal`

File names keep the flat op name for now, for example
`generated/base/special_erfinv.h`.

In-place ATen variants are normalized by the generator so they do not collide
with non-in-place public operator names. Treat them carefully: they may need
operator-specific tests because ATen's mutation semantics are not always a good
fit for InfiniOps' explicit-output API.

## Common skip reasons

The generator prints skipped operators and reasons during configure. Common
causes include:

- `no .out variant`: the ATen op has no usable `_out` form.
- `unsupported ATen type`: the schema uses a type that is not mapped to an
  InfiniOps-facing C++ type.
- `no testable tensor input/output pair`: the op does not expose both tensor
  inputs and tensor outputs in a form the generator can validate.
- `duplicate visible C++ signature`: two ATen overloads collapse to the same
  public C++ signature after hidden defaults; one is kept.
- `existing base has no overload compatible with ATen schema`: a hand-written
  `src/base/<op>.h` exists, but none of its overloads match a usable ATen
  schema.

Do not ignore these warnings when adding a new operator. A skipped operator is
not available through the PyTorch backend even if it remains listed in
`scripts/torch_ops.yaml`.

## Generated backend testing checklist

For a generated ATen-backed operator, validate at least:

1. The generator succeeds and emits the expected files.
2. The focused smoke build succeeds with `WITH_TORCH=ON` and the target
   platform backend enabled.
3. `tests/test_torch_ops.py` includes the operator and passes on CPU plus the
   target platform.
4. A focused handwritten operator test passes across supported platforms when
   the operator is part of the normal operator test set.
5. `ruff format --check`, `ruff check`, and any touched C++ formatting checks
   pass.

When an operator is unsupported by a vendor PyTorch fork, keep the skip local
and explicit in the operator test. If the vendor backend aborts or segfaults,
do not hide it in a broad generated test result; split the operator into a
follow-up and document the failing backend and reason.

## Add a hand-written ATen backend

A hand-written ATen backend is just a normal InfiniOps backend implementation
that happens to call PyTorch C++ APIs. It is compiled when `WITH_TORCH=ON`.

Use this path when generation is a poor fit:

- The operator is not an ATen op name, for example `Gemm`.
- The public InfiniOps API differs from ATen's `.out` schema.
- The implementation needs custom control flow, such as transposition,
  batching, alpha/beta handling, or using different ATen calls per device.
- ATen has no usable `.out` form, but a functional ATen call can be copied into
  an explicit output tensor.

### Files to add

For an operator named `Foo`, add:

- `src/base/foo.h`: the public InfiniOps operator base, if it does not already
  exist.
- `src/torch/ops/foo/foo.h`: the PyTorch backend declaration.
- `src/torch/ops/foo/foo.cc`: the PyTorch backend implementation.
- A focused test under `tests/test_foo.py`.

The source files under `src/torch/` are picked up by CMake when
`WITH_TORCH=ON`. The binding generator scans `src/torch` as an active
implementation root, so a smoke build should include the op in
`INFINI_OPS_OPS` if the smoke allowlist is active.

You do not need to add the op to `scripts/torch_ops.yaml` for a hand-written
backend. Only add it there if you also want the generated backend/metadata path
and have checked that it does not collide with the hand-written implementation.

### Backend declaration

Declare a specialization of `Operator<Foo, kDev, kIndex>` for all devices
represented by `kDev`.

Example shape, based on `src/torch/ops/add/add.h`:

```cpp
#ifndef INFINI_OPS_TORCH_FOO_H_
#define INFINI_OPS_TORCH_FOO_H_

#include "base/foo.h"

namespace infini::ops {

template <Device::Type kDev>
class Operator<Foo, kDev, kIndex> : public Foo {
 public:
  Operator(const Tensor input, Tensor out);

  void operator()(const Tensor input, Tensor out) const override;

 private:
  int device_index_{0};
};

}  // namespace infini::ops

#endif
```

Choose `kIndex` deliberately:

- Generated ATen wrappers reserve index `8`.
- Existing hand-written ATen examples use explicit non-generated indices:
  `Add` uses `1`, and `Gemm` uses `2`.
- Do not reuse an index already used by the same operator for a native/vendor
  backend.

If the base class has multiple overloads and the PyTorch backend overrides only
one of them, add `using Foo::operator();` in the derived class so the inherited
overloads remain visible. `Gemm` uses this pattern.

### Backend implementation

Include the backend header and `src/torch/tensor_.h`. The tensor helper converts
InfiniOps tensor metadata into `at::Tensor` views:

```cpp
#include "torch/ops/foo/foo.h"

#include "torch/tensor_.h"

namespace infini::ops {

template <Device::Type kDev>
Operator<Foo, kDev, kIndex>::Operator(const Tensor input, Tensor out)
    : Foo{input, out}, device_index_{out.device().index()} {}

template <Device::Type kDev>
void Operator<Foo, kDev, kIndex>::operator()(const Tensor input,
                                             Tensor out) const {
  auto at_input =
      ToAtenTensor<kDev>(const_cast<void*>(input.data()), input_shape_,
                         input_strides_, input_type_, device_index_);
  auto at_out = ToAtenTensor<kDev>(out.data(), out_shape_, out_strides_,
                                   out_type_, device_index_);

  at::foo_out(at_out, at_input);
}

template class Operator<Foo, Device::Type::kCpu, kIndex>;
template class Operator<Foo, Device::Type::kNvidia, kIndex>;
template class Operator<Foo, Device::Type::kCambricon, kIndex>;
template class Operator<Foo, Device::Type::kAscend, kIndex>;
template class Operator<Foo, Device::Type::kMetax, kIndex>;
template class Operator<Foo, Device::Type::kMoore, kIndex>;
template class Operator<Foo, Device::Type::kIluvatar, kIndex>;
template class Operator<Foo, Device::Type::kHygon, kIndex>;

}  // namespace infini::ops
```

Use metadata cached in the base constructor, such as `input_shape_` and
`input_strides_`, when calling `ToAtenTensor`. Do not rely on reading shape or
stride data from the runtime `Tensor` argument in the call path; it may have
been moved by dispatch machinery.

For input tensors, `ToAtenTensor` currently takes `void*`, so existing
hand-written backends use `const_cast<void*>(input.data())`. Output tensors pass
`out.data()` directly.

### When one ATen call is not enough

The hand-written backend can contain arbitrary ATen-side logic. `Gemm` is the
main example:

- It converts `a`, `b`, and `c` into ATen tensor views.
- It applies `trans_a` and `trans_b` by calling `transpose`.
- It uses `addmm_out` / `baddbmm_out` on CPU and NVIDIA.
- It falls back to `matmul` plus `copy_`, `mul_`, and `add_` on other devices.

This style is appropriate when InfiniOps semantics are stable but the best ATen
implementation varies by rank, device, or parameter values.

### Build and test hand-written ATen backends

For a focused smoke build:

```bash
python3 -m pip install --no-build-isolation --no-deps . \
  --config-settings=cmake.define.WITH_CPU=ON \
  --config-settings=cmake.define.WITH_TORCH=ON \
  --config-settings=cmake.define.INFINI_OPS_SMOKE_BUILD=ON \
  --config-settings=cmake.define.INFINI_OPS_OPS=foo
```

Add the target platform flag as needed, such as `WITH_NVIDIA=ON` or
`WITH_CAMBRICON=ON`.

Then run the focused operator test:

```bash
python3 -m pytest tests/test_foo.py -q -v --devices cpu <platform>
```

For hand-written ATen backends, a focused operator test is required. The
generated `tests/test_torch_ops.py` coverage is driven by
`generated/torch_ops_metadata.json`, so it only covers generated ATen wrappers.
