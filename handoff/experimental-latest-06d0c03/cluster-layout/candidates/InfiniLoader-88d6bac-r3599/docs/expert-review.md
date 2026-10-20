# Independent expert review

Date: 2026-10-18 (Asia/Shanghai). Base: upstream master `5014a6a`.
Branch: `codex/reliability-usability`. No merge into master was performed.

Three independent reviewer agents inspected the complete original nine-commit range,
with complementary specializations. They used read-only source review and isolated
Linux probes. Implementation and fixes were performed by the primary agent, then
returned to the reviewers for independent verification.

## Findings and resolution

| Priority | Finding | Resolution and regression evidence |
| --- | --- | --- |
| P1 | IPC timeout could signal a ready daemon despite active leases | Signal fallback now requires genuine pre-ready state or explicit force. Real saturated-socket regression preserves the leased daemon. |
| P1 | Custom runtime paths could replace ordinary files, symlinks or other Unix services | Preflight validates resource types, marked stale state, owner and endpoint identity. Existing listeners are refused. Atomic metadata writes and inode claims prevent cleanup of replaced resources. |
| P2 | Verification accepted extra unverified model files | Full inventory must match the manifest, including duplicate/source-name checks. Extra safetensors cause invalidation and a new generation. |
| P2 | Malformed manifest records prevented verification and refresh | Schema failures return invalid status. Timestamp overflow is normalized; totals must be non-boolean integers matching recorded sizes. GC measures actual file bytes. |
| P2 | Copy buffers could exceed the advertised reservation | Byte buffers and memoryviews are released before the next read. Prefix hashing is chunked. Tracemalloc regressions check both memfd and disk copy paths. |
| P2 | Bulk stop aborted after the first leased cache | Per-instance errors are reported while subsequent instances are processed. A two-daemon regression verifies continuation. |
| P2 | Distributed failure text omitted synthetic rank paths and raised KeyError | Every cancellation row includes its rank path. Text-mode failure diagnostics are covered. |

Follow-up fixes are deliberately separate commits:

- `7e1b248`: runtime resource ownership, leased stop fallback, bulk stop and rank diagnostics.
- `cc67790`: generation integrity, malformed manifests, GC accounting and bounded copy buffers.

## Per-commit independence

The CLI/validation expert exported each original commit to a fresh temporary archive
and ran its complete test suite, independently of the working checkout. All passed
on Linux / Python 3.14.4:

| Commit | Passing tests |
| --- | ---: |
| `2f7204b` | 27 |
| `c67ba9e` | 31 |
| `deaa940` | 34 |
| `d726100` | 39 |
| `2f54480` | 46 |
| `fb2f057` | 51 |
| `8fab6cd` | 55 |
| `a17a6e9` | 55 |
| `6a33d83` | 55 |
| `7e1b248` | 66 |
| `cc67790` | 77 |

## Final independent verdicts

- Lifecycle expert approved `origin/master..cc67790` within daemon lifecycle, IPC,
  path/resource ownership and distributed-cancellation scope. The updated targeted
  set passed 44 tests. Additional probes verified ordinary-file/listener preservation,
  lease timeout protection, and recovery after SIGKILL of an owned daemon.
- Cache/memory expert approved `origin/master..cc67790` within cache integrity,
  generation publication, leases/GC and memory accounting scope. The final independent
  targeted set passed 28 relevant tests; all original and follow-up findings resolved.
- CLI/validation expert confirmed the three CLI findings resolved, with 28 targeted
  tests passing, and independently verified each commit's full suite as above.
  Final approval is tied to `cc67790f7b4d65b20cbeef051f5ee9fc6189088c`; its isolated
  complete suite passed 77 tests in 19.88 seconds (preceding runtime fix: 66 tests).

The implementation is considered merge-ready within this branch's agreed reliability
and local-usability scope once its final remote CI is green. This is a code-review
judgement, not an approval to merge. Native shard conversion and multi-node features
remain deferred. Real inference-engine/GPU compatibility, cross-container sharing
and performance are outside these CPU/file-service validations.
