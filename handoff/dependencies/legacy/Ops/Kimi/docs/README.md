# InfiniOps Documentation

InfiniOps is a high-performance operator library for InfiniCore. It provides
Python bindings and C++ operator implementations across CPU, NVIDIA, Iluvatar,
Hygon, MetaX, Cambricon, Moore, Ascend, and other backends.

The common public include entry is:

```cpp
#include <infini/ops.h>
```

The Python package entry is:

```python
import infini.ops
```

Start with these pages:

- [Getting Started](getting-started.md): install InfiniRT, build InfiniOps, and
  run a minimal operator call.
- [Build and Test](build.md): build options, smoke builds, and test commands.
- [Backends](backends.md): supported backend options and backend-specific
  requirements.
- [Examples](examples.md): Python and C++ examples currently available in the
  repository.
- [Core Types](api/core-types.md): tensors, devices, data types, handles, and
  configuration objects.
- [Operators](api/operators.md): the operator API model, dispatch, caching, and
  implementation layout.
- [Operator API Alignment](operator-api-alignment.md): select an upstream API,
  design canonical signatures, and review names, parameters, overloads, and
  return contracts.
- [API Reference](api/reference.md): generate the Doxygen C++ API reference.
- [Adding ATen-backed operators](aten-operators.md): generated and hand-written
  PyTorch ATen backend guidance.
- [Adding linked operators](linked-operators.md): adapt typed symbols from
  installed third-party shared libraries.
- [Compatibility](compatibility.md): public API boundary, generated files, and
  internal implementation headers.

The API reference can be generated locally with Doxygen. The Documentation Pages
workflow validates that generated HTML for pull requests and publishes `master`
builds through GitHub Pages.
