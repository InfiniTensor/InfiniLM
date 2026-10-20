# Operators

InfiniOps operators are C++ classes with generated Python bindings. They share
a common dispatch model across devices and backend implementations.

## Public Call Shape

Python users call generated functions from `infini.ops`:

```python
import infini.ops

infini.ops.gemm(a, b, out)
```

C++ users call documented operator classes:

```cpp
infini::ops::Gemm::Call(a, b, out);
```

For calls that need a stream, workspace, or implementation selection, pass
`Handle` and `Config` explicitly:

```cpp
infini::ops::Handle handle;
infini::ops::Config config;

config.set_implementation_index(1);
infini::ops::Gemm::Call(handle, config, a, b, out);
```

## Dispatch Model

Each operator has:

- a base class under `src/base/<op>.h`
- zero or more native implementations under `src/native/.../ops/<op>/`
- optional PyTorch C++ implementations under `src/torch/ops/<op>/`
- generated wrappers and Python bindings under `generated/`
- tests under `tests/test_<op>.py`

`Operator<Key, Device, Index>` specializations provide concrete
implementations. `Device` selects the backend and `Index` selects the
implementation slot for that operator on that backend.

## Implementation Indexes

Implementation indexes are local to an operator and device. Use an explicit
index only when the backend exposes multiple implementations for the same
operator.

Generated ATen-backed wrappers reserve implementation index `8`. Hand-written
backend implementations must avoid colliding with existing implementations for
the same operator.

## Operator Cache

`Operator::Call(...)` caches constructed operator instances per thread. The
cache key includes the config implementation index and tensor/scalar geometry.
Tests can call `clear_cache()` on generated Python operator classes for module
isolation.

Code that changes tensor geometry, backend implementation selection, or
workspace assumptions should account for this caching behavior.

## Tracing Calls

Set `INFINI_OPS_TRACE_CALLS=1` to print each call that reaches the InfiniOps
dispatcher. Each line starts with the triggering environment variable and is
followed by a JSON object containing the operator, device, and implementation
index:

```text
[INFINI_OPS_TRACE_CALLS] {"operator_name": "Gemm", "device_type": "iluvatar", "implementation": 0}
```

Tracing is disabled by default.

## Adding an Operator

The standard path for a native operator is:

1. Add the base class in `src/base/<op>.h`.
2. Add one or more backend implementations under `src/native/.../ops/<op>/`.
3. Add or update generated wrapper inputs if needed.
4. Add focused tests under `tests/test_<op>.py`.
5. Validate a smoke build plus the focused test on every affected backend.

Before defining the public base interface, follow
[Operator API Alignment](../operator-api-alignment.md) to select an upstream
target and design the canonical name, parameter list, overloads, and return
contract.

For PyTorch ATen-backed operators, see
[Adding ATen-backed operators](../aten-operators.md). That page explains the
generated backend path and the hand-written ATen backend path.

## Smoke Coverage

Smoke builds use an operator allowlist to keep routine validation short:

```bash
python -m pip install .[dev] --no-build-isolation --no-deps \
  --config-settings=cmake.define.INFINI_RT_ROOT=/path/to/infini-rt-prefix \
  --config-settings=cmake.define.WITH_CPU=ON \
  --config-settings=cmake.define.INFINI_OPS_SMOKE_BUILD=ON
```

Then run:

```bash
python -m pytest tests -m smoke -q
```

Use full builds and broader tests for shared dispatch, wrapper generation,
backend infrastructure, or high-risk operator changes.
