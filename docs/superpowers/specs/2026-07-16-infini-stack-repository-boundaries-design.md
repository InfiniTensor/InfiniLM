# Infini Stack Repository Boundaries Design

## Status

Approved in conversation on 2026-07-16. This document is the implementation
contract for the NVIDIA migration slice.

## Goal

Make InfiniCore a versioned manifest for InfiniRT, InfiniOps, and InfiniCCL,
with no legacy runtime API or build ownership. Move the integration build and
runtime-facing ownership into InfiniLM, and update the shared InfiniLM CI to use
that new boundary.

The migration is intentionally a clean break. InfiniCore must not retain
compatibility definitions such as `infiniStatus_t`, legacy `infiniop` or
`infinirt` source trees, forwarding headers, Python packaging, or a second build
system.

## Confirmed NVIDIA Support Boundary

The current submodule revisions are sufficient for the first migration target,
but this does not imply complete inference support on every model or backend.

- InfiniRT covers the device, memory, stream, event, and graph lifecycle used by
  the migrated InfiniLM runtime on NVIDIA.
- InfiniOps provides the 23-operation allowlist used by the current dense,
  non-quantized Qwen3 path with the default static attention implementation.
- InfiniCCL provides the initialization, destruction, and AllReduce behavior
  required by dense Qwen3 tensor parallelism.
- The validated end-to-end baseline is NVIDIA, Qwen3, tensor parallel size 1,
  without quantization. Tensor parallel size 2 is a required acceptance test for
  this cleanup.

The following remain outside this slice: other accelerator backends, complete
model-family coverage, quantized paths, FlashAttention or FlashInfer paths, and
MoE expert parallelism. In particular, InfiniCCL's NVIDIA backend does not yet
provide the AllGather and ReduceScatter operations used by the InfiniLM MoE
expert-parallel path. Those additions require separate feature work and must not
be implied by this refactor.

## Repository Ownership

| Repository | Responsibility after the migration |
| --- | --- |
| InfiniRT | Runtime primitives and public runtime types |
| InfiniOps | Operator implementations and public operator API |
| InfiniCCL | Collective communication API and backends |
| InfiniCore | Pins the three repositories above as git submodules |
| InfiniLM | Builds the pinned stack and owns inference/runtime integration |
| InfiniTensor/ci | Provides the reusable InfiniLM build and test workflow |

No runtime source, compatibility API, package metadata, tests, scripts, or CI
workflow belongs in InfiniCore after this change.

## Target InfiniCore Tree

InfiniCore will contain exactly these six tracked entries:

```text
.gitmodules
LICENSE
README.md
submodules/InfiniCCL
submodules/InfiniOps
submodules/InfiniRT
```

The three submodule gitlinks are the version contract. `.gitmodules` will only
define each path and URL; `branch` hints will be removed because they do not
control the pinned revision. The README will describe InfiniCore as a manifest,
show recursive checkout, and point integration users to InfiniLM. It will not
contain build instructions of its own.

The existing issue templates, workflow, `.gitignore`, `DEV.md`, `pyproject.toml`,
scripts, and tests will be deleted from InfiniCore. Build integration content is
migrated as described below; repository-specific legacy metadata is not copied.

## InfiniLM Build Interface

InfiniLM will own these files:

```text
scripts/build_infini_stack.py
scripts/configs/infiniops_ops.txt
test/scripts/test_build_infini_stack.py
```

`scripts/build_infini_stack.py` will preserve the existing focused NVIDIA build
behavior and add a required `--infinicore-root PATH` argument. The argument
points to a separately checked-out InfiniCore manifest. The script will:

1. Resolve the InfiniCore checkout and verify its three submodules are
   initialized at the gitlink revisions recorded in InfiniCore `HEAD`.
2. Read the operator allowlist from the InfiniLM-owned configuration file.
3. Build and install InfiniRT, InfiniOps, and InfiniCCL into one isolated prefix.
4. Keep source checkouts immutable and place all generated state below the
   selected build root.
5. Write a manifest containing the InfiniLM revision, InfiniCore revision, all
   three submodule revisions, build options, install prefix, and operator list.

The existing `--build-root`, `--build-type`, `--cuda-arch`, `--jobs`, `--test`,
and `--dry-run` behavior remains. Relative build roots are resolved against the
InfiniLM checkout. `--test` continues to enable InfiniRT tests and the two-GPU
InfiniCCL AllReduce smoke test.

InfiniLM's README and contributing guidance will identify this script as the
supported way to build the pinned native stack. InfiniLM's own `third_party`
submodules remain the source of its JSON and spdlog dependencies.

## Shared CI Contract

The reusable workflow currently consumed from
`InfiniTensor/ci/.github/workflows/infinilm-ci.yml@infiniCore_ci` must change in
the same integration series. Work will branch from the currently referenced
`infiniCore_ci` integration line and follow that repository's contribution
rules.

The workflow will:

1. Check out InfiniLM with its own submodules, including JSON and spdlog.
2. Check out the selected InfiniCore revision recursively to obtain the three
   pinned stack repositories.
3. Invoke InfiniLM's `scripts/build_infini_stack.py --infinicore-root ...`.
4. Build and test InfiniLM against the resulting install prefix.

The workflow will stop reading `third_party` content from InfiniCore, stop
building InfiniCore with xmake, and stop installing InfiniCore as a Python
package. Existing branch/ref inputs may be retained when they still express the
same checkout selection; unnecessary interface renames are avoided.

During cross-repository validation, the InfiniLM caller will temporarily pin the
feature revision of the shared workflow. It will return to the canonical shared
CI revision after that workflow change is integrated.

## InfiniCCL Pull Request

The InfiniCCL change remains a standalone header-correctness fix:

- include `<cstdint>` in `src/device.h`;
- use `std::uint8_t` as the `MemorySpace` underlying type.

The branch will be rebased onto the current InfiniCCL `master`, formatted and
checked according to that repository's `CONTRIBUTING.md`, and submitted as a
ready pull request using its current pull request template. Validation evidence
will include the NVIDIA/NCCL examples and an InfiniLM Qwen3 tensor-parallel-size
2 run. The pull request will not claim to add collectives or broaden backend
support.

For integration validation, InfiniCore may temporarily pin the rebased pull
request head. The InfiniCore cleanup is not merge-ready until the InfiniCCL pull
request is merged and the gitlink points to its durable commit on `master`.

## Change Sequence And Commits

Each repository keeps its own conventions and focused commits.

1. Commit this design in InfiniLM.
2. Prepare the shared CI branch and confirm its repository-specific rules.
3. Move the build script, operator configuration, tests, and documentation into
   InfiniLM in focused migration and documentation commits.
4. Update the reusable shared CI workflow in a separate CI repository commit.
5. Reduce InfiniCore to the six-entry manifest in focused migration, metadata,
   and cleanup commits as needed to keep each commit reviewable.
6. Rebase, verify, push, and open the standalone InfiniCCL pull request.
7. Validate clean coordinated checkouts on NVIDIA before presenting any branch
   as merge-ready.

Cross-repository branches are allowed to reference one another during
validation. Final merge commits must use durable branch or merged revisions,
not an unreviewed local checkout.

## Verification

Local and structural checks:

- run each repository's formatter and static checks before every commit;
- run the migrated Python unit tests, including gitlink mismatch, missing
  submodule, invalid operator list, argument validation, manifest, and dry-run
  command coverage;
- verify `git ls-files` in InfiniCore produces exactly the six target entries;
- verify no tracked InfiniCore file contains legacy runtime identifiers;
- validate workflow syntax and run `actionlint` when available;
- inspect each repository diff against its intended base before push.

NVIDIA acceptance checks in `accelerator-dev/nvidia:latest`:

- start from clean coordinated checkouts with recursively initialized
  submodules;
- run the InfiniLM-owned stack builder with tests enabled;
- run InfiniRT lifecycle and graph smoke coverage;
- confirm all 23 selected InfiniOps adapters build and link;
- run the InfiniCCL NVIDIA/NCCL AllReduce example;
- build and install the InfiniLM wheel against the generated prefix;
- run the existing Qwen3 tensor-parallel-size 1 inference command;
- run the same supported Qwen3 path with tensor parallel size 2;
- repeat the final inference smoke in a fresh container to prove it does not
  depend on untracked host artifacts.

The pull request descriptions will distinguish these verified paths from the
explicitly excluded support areas.

## Risks And Recovery

- Removing InfiniCore's build and package surfaces is intentionally breaking for
  old consumers. The supported recovery is migration to InfiniLM's build script,
  not compatibility shims in InfiniCore.
- Shared CI and product repositories cannot merge atomically. Feature refs will
  be used for validation, and merge order will keep every canonical ref usable.
- A rebased or merged InfiniCCL pull request changes its commit identity. The
  InfiniCore gitlink and validation manifest must be refreshed together.
- NVIDIA success does not authorize enabling other model or accelerator paths.
  Existing feature gates remain until those paths receive their own validation.
- If coordinated NVIDIA validation regresses, retain the prior pushed branches
  and fix forward in the owning repository; do not restore legacy runtime code
  to InfiniCore.
