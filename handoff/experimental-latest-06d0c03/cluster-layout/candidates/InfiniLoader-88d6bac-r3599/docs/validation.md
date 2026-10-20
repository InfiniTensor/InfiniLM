# Reliability and usability validation

Date: 2026-10-18 (Asia/Shanghai). Baseline: `5014a6a` on upstream master.

## Local results

- Ubuntu on WSL2, Linux 6.18.33.2, Python 3.14.4: **77 tests passed** after independent
  review fixes (25.92 seconds). The baseline contained 17 tests; the original nine
  implementation commits ended with 55 passing tests.
- Built `infiniloader-0.1.0-py3-none-any.whl` with Python 3.12.14/setuptools 84.0.0.
  The package version is unchanged; this branch has not been published as a release.
- Executed `python -m infiniloader selftest` from the built wheel on Linux: passed
  the disk-cache reuse, sealed memfd view, child-process open and mmap checks.
- Executed `doctor` from the built wheel: Linux, advisory locks, Unix sockets,
  memfd seals and cross-process proc reopening all passed.
- `git diff --check`: passed.

## Regression coverage

Real child processes cover concurrent CLI starts, joining a daemon during preload,
model/revision mismatch rejection, lease-protected commands, idle expiry, safe
resident eviction, and stalled/disconnected IPC clients. Other tests check sealed
weights, FD cleanup, mutation during preload, unsafe path rejection, disk corruption,
SHA-256 verification, immutable refresh, LRU/TTL and leases, cgroup v1/v2 ancestor
limits, pending resident reservations, and failed-rank cancellation and reaping.

The idle-expiry integration test uses a two-second TTL so process startup scheduling
does not consume an unrealistically short test deadline. The test waits beyond the
TTL while a lease is active, then verifies eviction after lease release.

## Scope and remaining validation

The original branch HEAD `6a33d83` passed the complete remote Ubuntu Python 3.10-3.14
matrix in [Actions run 39101870656](https://github.com/InfiniTensor/InfiniLoader/actions/runs/39101870656).
Check the newer branch workflow for the subsequent review fixes. Independent review
and per-commit test details are recorded in [expert-review.md](expert-review.md).
The Dockerfile is a non-root diagnostic image; Docker was unavailable locally, so
the image was not built or exercised in this environment.

These are CPU/file-service tests using synthetic model files. They do not establish
vLLM/SGLang/InfiniLM version compatibility, GPU transfer throughput, real TP/PP loading,
cross-container namespace sharing, or multi-node performance. Existing README timing
numbers are historical. Native shard conversion and multi-node coordination are
deferred to subsequent work as agreed for this branch.
