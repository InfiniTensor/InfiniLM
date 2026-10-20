# API Reference

InfiniOps keeps user-facing documentation in Markdown and can generate a C++
API reference with Doxygen.

## Generate Reference HTML

Configure a build with documentation enabled:

```bash
cmake -S . -B build \
  -DINFINI_RT_ROOT=/path/to/infini-rt-prefix \
  -DWITH_CPU=ON \
  -DINFINI_OPS_BUILD_DOCS=ON
```

Then generate the HTML reference:

```bash
cmake --build build --target infiniops_docs
```

The generated HTML is written under:

```text
build/docs/reference/html
```

## Preview the Rendered Structure

Serve the generated HTML directory with a local static file server:

```bash
python -m http.server 8000 --directory build/docs/reference/html
```

Then open:

```text
http://localhost:8000/
```

The Doxygen page includes a left-side tree view and search box for browsing the
rendered API structure.

## Theme Assets

The generated reference uses Doxygen Awesome through CMake `FetchContent` plus a
small InfiniOps-specific stylesheet under `docs/assets`. The theme source is
downloaded at configure time for documentation builds and is not checked into
this repository, so configuring with `INFINI_OPS_BUILD_DOCS=ON` requires network
access unless the dependency is already available in the CMake cache.

## Reference Scope

The Doxygen configuration is intentionally scoped to the public include entry,
generated public operator headers, core operator infrastructure, base operator
classes, and the Markdown documentation under `docs/`.

The following are not intended as the primary user documentation surface:

- backend implementation details under `src/native/`
- PyTorch backend implementation details under `src/torch/`
- generated implementation sources under `generated/src`
- generated Python binding sources under `generated/bindings`

Use [Compatibility](../compatibility.md) for the supported API boundary.

## Publishing

The Documentation Pages workflow builds the same `infiniops_docs` target for
pull requests and uploads the generated HTML as a Pages artifact. Pushes to
`master` deploy the artifact through GitHub Pages when the repository Pages
source is configured for GitHub Actions.
