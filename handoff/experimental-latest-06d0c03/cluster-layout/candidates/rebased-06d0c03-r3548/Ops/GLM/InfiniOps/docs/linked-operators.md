# Linked Operators

The linked backend calls operators provided by an installed third-party shared
library. It supports exact exported symbols, TVM FFI entry points, and
registered PyTorch Dispatcher operators when a platform package does not
provide source code or a stable C API.

## Source Layout

Linked implementations use the following layout:

```text
src/linked/<transport>/<device>/
  <library>.yaml
  ops/<operator>/
    <implementation>.yaml
    <implementation>.h
    <implementation>.{cc,cu}
```

CUDA providers may use `<implementation>.cu` instead of `.cc`.

The platform library file contains DSO discovery information:

```yaml
python_distribution_package: vllm
library_glob: vllm/_C*.so
```

A library may also provide `include_glob` when its transport needs installed
headers. Each glob must resolve to exactly one path in the Python distribution.
A library may set `python_distribution_version` to a PEP 440 specifier. The
resolver verifies the installed distribution version before looking up its DSO.
A DSO that depends on another declared platform library lists that dependency
under the implementation's optional `link_libraries` key.

Files for an operator implementation use the provider name as their common
stem. Multiple implementations for the same operator and device use distinct
file stems and implementation slots.

The implementation YAML identifies the library and the exact demangled symbols
required by its C++ source:

```yaml
library: vllm
required_symbols:
  - silu_and_mul(at::Tensor&, at::Tensor&)
```

A Dispatcher implementation instead declares its exact schema and required
dispatch key:

```yaml
library: vllm
operator_schema: >-
  _C::gptq_marlin_repack(Tensor b_q_weight, Tensor perm, SymInt size_k,
  SymInt size_n, int num_bits, bool is_a_8bit) -> Tensor
dispatch_key: CUDA
```

Each implementation uses exactly one contract form. The resolver rejects a
partial Dispatcher contract or a binding that mixes both forms.

## Adapter Boundary

Keep ABI behavior in `<implementation>.cc` or `.cu`, not in YAML. Shared
operator templates own reusable tensor conversion, stream guards, layout
staging, and copy-back behavior. Provider sources own exact typed declarations,
synthesized arguments, and provider-specific return handling.

For the `torch` transport, an implementation backend inherits its device `C10`
specialization for device identity and external-stream handling, then defines
its provider-specific `Call` ABI.

For the `tvm_ffi` transport, provider sources call exported TVM FFI entry
points directly. The resolver supplies installed TVM FFI headers and links the
provider DSO together with every library named in `link_libraries`.

## Configuration

At configure time, `scripts/resolve_linked_ops.py` locates the installed Python
distribution. It verifies C++ symbols with both `nm` and `readelf`; raw
`readelf` symbols are demangled with GNU or LLVM `c++filt` before exact
comparison. Dispatcher contracts are verified in an isolated Python process by
loading the DSO, comparing the registered schema exactly, and checking the
requested dispatch key. The resolver writes a CMake manifest and diagnostic
JSON under `generated/linked/`. Generated files are not committed.

DSOs that provide Dispatcher registrations and DSOs named in
`link_libraries` are retained with `--no-as-needed` only for their own link
items. This prevents the linker from discarding registration code or an
explicit dependency while preserving its state for unrelated libraries.

Enable the backend independently of generated ATen implementations:

```bash
cmake -S . -B build \
  -DWITH_METAX=ON \
  -DWITH_LINKED=ON \
  -DWITH_TORCH=OFF \
  -DINFINI_OPS_OPS=silu_and_mul
```

`WITH_TORCH=OFF` disables the generated ATen backend, but `WITH_LINKED=ON`
currently still requires an installed `torch` package. Linked configuration
shares its Python interpreter and C++ ABI setup with the existing `torch`
transport.

To resolve only selected linked implementation slots, pass an `ops.json` file
through `INFINI_OPS_OPS`. The resolver reads each linked provider's slot from
its sibling C++ header before locating external libraries, so an unselected
provider does not add a package or shared-library dependency. See
[Build configuration](build.md) for the file format.

The `torch` transport uses the installed PyTorch C++ headers and libraries for
`at::Tensor`, but it does not enable the standard `src/torch` operator backend.
Provider and PyTorch C++ ABIs must match. Configuration fails before compilation
when the distribution, shared library, or an exact required symbol is missing.

InfiniOps does not bundle the provider library. Its resolved directory and the
directories of its linked dependencies are recorded in the installed binary's
RPATH, so a linked build is tied to that Python environment. PyTorch runtime
directories are recorded for the `torch` transport as well. Reconfigure and
rebuild after moving or replacing the provider environment. In-place changes
to a resolved provider DSO are tracked as CMake configure and link dependencies.

## Implementation Slots

A linked provider declares an implementation slot of `16` or greater in its
`Operator` specialization. The slot must be unique for that operator and
device.
