# Common CoreX validation adapter

These files record the common build adapter used for the official and combined
PR validation. They are validation inputs, not default installed providers.
Both roles use exactly the same adapter.

Set both `CMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0` and
`TORCH_CXX11_ABI=0` for this SDK. The latter also supplies the ABI flag to
the custom CoreX dispatch commands when `WITH_TORCH=OFF`. The existing
InfiniLM/InfiniRT libraries use ABI0; mixing ABI1 string factories with that
SDK produces unresolved symbols even after a successful shared-library link.

The tested upstream CoreX build does not register the existing generic CUDA
Argmax, Triu and Tril implementations. Copy the three thin wrappers into
`src/native/cuda/iluvatar/ops/{argmax,triu,tril}/kernel.h` in each disposable
InfiniOps source copy before CMake generates its dispatch units. The wrappers
reuse the existing kernels and runtime; they introduce no new arithmetic.
`ops_selection.json` fixes the 72 operators selected for the Llama SDK and
submitted provider regressions.

The monolithic InfiniLM extension also instantiates unused MoE/LayerNorm paths
whose providers are unavailable in this CoreX configuration. Its validation
build force-includes a generated header in this order:

1. `<infini/ops.h>` from the current generated include directory.
2. `base/<operator>.h` for every operator in `ops_selection.json`.
3. `corex_unavailable_factory.h`.

The primary factory returns `CanMake=false` and throws on `Make`/`Call`.
Available generated specializations retain their original dispatch. A host
probe against the actual generated headers verifies all three unavailable
factory outcomes. The unused operator classes also need their original
`Operator<Key>::DefaultConfig(Device::Type)` definitions instantiated from
`src/operator.h`. Include that implementation before the corresponding base
headers and instantiate only the unavailable types; do not add provider
registrations. The original empty-provider exception remains intact. Check
both CPU and Iluvatar requests reject, then import both complete SDKs before
starting any GPU work.

Both execution SDKs use writable copy-on-write safetensors mappings
(`mmap.ACCESS_COPY`) and mark the returned NumPy views read-only with
`array.setflags(write=False)`; the source weight files remain unchanged.
This preserves the previously validated loader ownership contract. Automatic operator tuning
is explicitly disabled in both roles with `INFINI_OPS_ENABLE_AUTO_TUNING=0`.

The build record retains the original source archive hashes, per-command
arguments, compiler depfiles, successful command receipts and library hashes.
Object reuse requires matching source, flags and every dependency byte, plus
an ordinary successful receipt. The device tests additionally exercise the
common Argmax and triangular-mask paths alongside the submitted optimizations.

The current Iluvatar FlashAttnVarlen provider implements causal paged inference.
Its inherited generic metadata predicate also accepts nonpaged static-cache
requests, which then abort at the native provider assertion. Apply
`corex_flash_varlen_compat.h` at
`src/native/cuda/iluvatar/ops/flash_attn_varlen_func/kernel.h` in both disposable
source copies. Its provider-local `supports` checks the existing paged-only
contract before dispatch; the native operator body and original nonpaged
InfiniLM fallback stay unchanged. The short interface explicitly rejects because
it has no block table. This common compatibility adapter is separate from the
four measured optimizations.
