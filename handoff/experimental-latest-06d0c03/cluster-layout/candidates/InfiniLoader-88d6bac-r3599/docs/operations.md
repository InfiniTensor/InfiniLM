# Operating the development version

This branch improves reliability and local usability. TP checkpoint conversion and
multi-node transport remain future work. Runtime dependencies remain limited to
Python's standard library. Linux and Python 3.10+ are required.

## Choose a backend

```bash
MODEL=/models/checkpoint
iloader doctor
VIEW=$(iloader start "$MODEL" --name dev --local-cache)
iloader run dev -- vllm serve '{model}' --tensor-parallel-size 4
```

`run` replaces literal `{model}` placeholders in arguments and sets
`ILOADER_MODEL_PATH`. It executes arguments without a shell and returns the command's
exit code. Put InfiniLoader options before the cache name. Use a foreground engine
command that remains alive while its workers run.

For disk-only caching without another model-sized RAM allocation:

```bash
LOCAL=$(iloader cache "$MODEL" --name dev)
iloader verify "$LOCAL"
iloader run --cache-path "$LOCAL" -- vllm serve '{model}'
```

The OS controls disk cache residency. Memfd files are sealed against writes, growth
and truncation, but their pages are not mlocked and can be swapped. Status reports
logical file capacity, not measured physical residency. Parsing, CPU tensor allocations,
weight transformation and CPU-to-device transfers remain the engine's responsibility.

## Identity, verification and migration

A resident name identifies one source path, file selection and source fingerprint.
The fingerprint includes sizes, nanosecond timestamps, device and inode IDs. Reusing
a name for another revision fails; stop the cache or choose another name. Changes
while copying also fail. Use immutable checkpoints: metadata checks cannot make an
external writer's multi-file update transactional.

Disk generations have version-2 manifests with per-file SHA-256 digests. Normal reuse
checks source and destination metadata; `verify PATH` reads all files and checks hashes.
`verify PATH --quick` only checks metadata. The digests describe copied bytes; they do
not authenticate a publisher or validate model semantics.

```bash
NEW_LOCAL=$(iloader cache "$MODEL" --name dev --refresh)
iloader verify "$NEW_LOCAL" --json
```

Refresh publishes a new generation, preserves old paths, and needs additional disk
space. SHA-256 is computed during the copy and its CPU cost belongs in preparation
benchmarks. Version-1 caches remain listed but are not reused; rebuild them, verify
the new path, and use `gc` to remove unleased old versions.

## Memory budgets

```bash
iloader start "$MODEL" --name dev --workers 4 \
  --max-memory-bytes 32GiB --reserve-bytes 4GiB --max-resident-bytes 64GiB
```

- Effective available memory is the minimum of host MemAvailable and remaining
  cgroup v1/v2 limits, including visible ancestors.
- The default budget is 90% of effective availability minus a 256 MiB reserve.
  Increase the reserve for engine CPU weights and other allocations.
- `--max-memory-bytes` caps one preload including estimated worker buffers.
- `--max-resident-bytes` sets a persistent quota for all daemons in the same runtime
  directory. A subsequent start can explicitly update the cap. In-flight loads reserve
  model and buffer bytes; ready caches reserve model bytes. Dead reservations are
  pruned using boot ID and PID start time.
- `--force` bypasses the available-memory check, while explicit per-cache and shared
  caps remain enforced. Quotas account for InfiniLoader, not engine allocations.
- Byte options accept integers and binary suffixes such as `256MiB` and `32GiB`.

Use a node-local `--base-dir`. A runtime registry shared between nodes is unsupported.

## Leases, LRU and TTL

`run` protects a cache against ordinary stop and eviction. Local-cache daemons also
hold a disk-generation lease. Foreground disk commands inherit its OS lock descriptor;
resident commands register the child process's PID and start time.

```bash
iloader gc --max-bytes 200GiB              # disk LRU preview
iloader gc --ttl 604800                   # disk generations idle for 7 days
iloader gc --max-bytes 200GiB --apply
iloader gc --resident --max-bytes 32GiB    # resident LRU preview
iloader gc --resident --ttl 3600 --apply
```

Without `--apply`, collection only previews. An unmet capacity target returns nonzero,
for example if leased caches exceed the target. Resident collection requests shutdown;
reported remaining capacity is projected until the daemons exit. Unmarked directories
and staging directories left by a killed writer are never automatically deleted.

```bash
iloader start "$MODEL" --name dev --idle-ttl 3600
iloader run dev -- vllm serve '{model}'
iloader stop dev --remove-view
```

Idle eviction is opt-in. Direct-path readers are not registered: keep idle eviction
disabled and avoid collecting their caches, or launch them through `run`. `stop --force`
overrides active resident leases. External signals also stop a leased daemon. Stale
PID metadata alone is never enough to signal a process.

## Diagnostics and containers

```bash
iloader doctor --json
iloader status dev --json
iloader list --json
iloader dist-simulate "$VIEW" --ranks 4 --timeout 60
```

Doctor exercises locks, Unix sockets, memfd sealing and reopening in another process.
It reports cgroup headroom, disk space, FD and memlock limits. Status includes identity,
file bytes, the startup budget snapshot, active leases, reuse requests and last use.
Distributed simulation is a bounded single-host read test, not a GPU benchmark.

Prefer installing InfiniLoader into the engine container and using the same UID.
Separate containers require the reader to see the daemon PID in its proc namespace
and pass proc and user-namespace access checks. A shared volume alone is insufficient.
Test an actual engine open in the deployment environment. Resolving individual view
file symlinks to their final target can break reopening; use the published file paths.
After a daemon exits, existing open descriptors/mappings may survive, but new opens
through its view fail. Automatic failover is not provided.

The non-root diagnostic image contains InfiniLoader, without an engine or model:

```bash
docker build -f deploy/Dockerfile -t infiniloader:dev .
docker run --rm --memory=2g infiniloader:dev
```

## Validation

Run `python -m pytest`, `iloader selftest`, and `iloader doctor`. CI tests Python
3.10-3.14 on Ubuntu. Tests exercise real daemon subprocesses, source mutations,
disconnected clients, immutable generations, quotas, leases, and failed ranks.

`bench` reports each run, first-run and median time, and best throughput. It does not
clear OS page cache. Compare cold/warm shared storage, cold/warm local disk, memfd and
engine-native loaders separately. Include preparation, end-to-end engine readiness,
peak CPU memory, per-rank bytes and CPU-to-device transfers. Historical README numbers
are not new measurements for this branch; CPU tests do not establish GPU or multi-node
performance.
