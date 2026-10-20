# Compatibility

This page defines the current documentation and compatibility boundary for
InfiniOps users and contributors.

## Public API Entries

The formal public entries are:

```cpp
#include <infini/ops.h>
```

```python
import infini.ops
```

InfiniOps exposes C++ as its formal native API and generated Python bindings as
its formal Python API. A separate C API is not currently exposed or planned.

The supported C++ compatibility boundary includes:

- the installed `<infini/ops.h>` entry point
- the core types documented under `docs/api/`
- the documented operator classes and `Call(...)` forms

The C++ boundary is intentionally narrower than the complete installed header
tree. A header being installed or included transitively does not by itself make
every symbol in that header part of the public API.

## InfiniRT Dependency

InfiniOps depends on InfiniRT headers and libraries from the configured
`INFINI_RT_ROOT`. Consumers should treat the installed InfiniOps wheel,
InfiniOps headers, and bundled or linked InfiniRT library as a matching set
from the same build.

## Generated Files

The build may generate files under `generated/`, including:

- public operator call instantiations
- generated base classes
- generated PyTorch C++ backend wrappers
- generated Python bindings
- generated metadata for tests

Generated files are build artifacts. Do not edit them by hand. Change the
source generator, allowlist, or source operator definitions instead.

## Implementation Details

Backend implementations under `src/native/**` and `src/torch/**`, generated
implementation sources, generated binding sources, and backend-specific runtime
adapters are implementation-facing. They are useful for contributors, tests,
and repository examples, but downstream users should not include them directly.

Support headers pulled in transitively by `<infini/ops.h>` compile the public C++
API. Undocumented helpers and templates in those headers remain implementation
details unless an API page explicitly includes them in the supported boundary.

## Backend-Specific Behavior

Operator availability and supported dtypes, layouts, strides, and implementation
indexes can differ by backend. Tests should document backend-specific skips or
tolerances explicitly instead of hiding them in broad generated results.

## Public API Compatibility

Follow these rules when changing public C++ and Python surfaces:

- Keep operator signatures stable unless the PR clearly documents the migration
  path.
- Preserve generated Python binding names when changing implementation details.
- Update documentation and tests in the same PR when user-visible behavior
  changes.
- Treat shared dispatch, wrapper generation, and backend selection changes as
  high-risk and validate them across affected platforms.
