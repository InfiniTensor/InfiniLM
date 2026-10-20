# InfiniRT Documentation

InfiniRT is the runtime layer of the InfiniCore project. This documentation
starts with the public runtime surface available today and is intended to grow
with lower-level backend capabilities and higher-level user-facing features as
they are added.

The supported public include entry is:

```cpp
#include <infini/rt.h>
```

Start with these pages:

- [Getting Started](getting-started.md): install InfiniRT and compile a minimal
  consumer.
- [Build and Test](build.md): CMake options for CPU, NVIDIA, and other
  backends.
- [Runtime API](api/runtime.md): the runtime namespace and backend dispatch
  model.
- [Core Types](api/core-types.md): `Device`, `DataType`, and `TensorView`.
- [Backends](backends.md): supported backend options and API support notes.
- [Compatibility](compatibility.md): stable API boundary and internal headers.
- [Contributing](../CONTRIBUTING.md): code style, commit conventions, PR
  workflow, development commands, and troubleshooting.

API reference generation is described in [API Reference](api/reference.md).
