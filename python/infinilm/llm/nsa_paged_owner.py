"""NSA-only immutable owner values, issued by the actual paged Scheduler.

No device readbacks or host-success completion inference. First domain excludes
connectors, speculative rewrite, multimodal, graphs and shared one-tail prefixes.
"""
from dataclasses import dataclass, replace
from threading import get_ident

U64_MAX = (1 << 64) - 1
I32_MAX = (1 << 31) - 1


def integer(value, low, high, label):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(label)
    return value


class OwnerClock:
    """One scheduler-owned counter; no identity registry.

    Bind on first mutation, since scheduler construction may precede its thread.
    pool_epoch is a required Root-issued/bound token, never id() or constant 1.
    A replacement Scheduler needs a new binding and invalidates old NSA state.
    """
    def __init__(self, pool_epoch):
        self.pool_epoch = integer(pool_epoch, 1, U64_MAX, "missing pool epoch")
        self._next = 1
        self._thread = None

    def issue(self, count=1):
        integer(count, 1, U64_MAX, "invalid issue count")
        thread = get_ident()
        if self._thread is not None and self._thread != thread:
            raise RuntimeError("foreign scheduler mutation")
        if self._next > U64_MAX - count + 1:
            raise OverflowError("paged owner token exhausted")
        first = self._next
        self._thread = thread
        self._next += count
        return tuple(range(first, first + count))


def claim_first(clock, request):
    """Call ONLY after existing _claim_request ownership checks, owner is None.

    Existing active-object claims must skip this hook. These fields are proposed
    declarations on InferenceRequest, not existing production members.
    """
    if request._nsa_incarnation != 0 or request._nsa_prefix_revision != 0:
        raise RuntimeError("request already issued")
    incarnation, revision = clock.issue(2)
    request._nsa_incarnation = incarnation
    request._nsa_prefix_revision = revision


def allocation_before_reset(clock, block):
    """Call after free-front/refcount checks but BEFORE popleft/reset/used.add.

    Only actual allocation issues this token. Pin/publish/free retaining a hash
    must not call it. Root adds allocation_generation = 0 in Block.__init__.
    """
    generation, = clock.issue()
    block.allocation_generation = generation


def revision_before_rewrite(clock, request):
    if request._nsa_incarnation == 0:
        raise RuntimeError("unclaimed request rewrite")
    revision, = clock.issue()
    request._nsa_prefix_revision = revision
    # A failed/partial rewrite stays invalidated. Never roll this token back.


@dataclass(frozen=True)
class Page:
    physical: int
    generation: int


@dataclass(frozen=True)
class OwnerRow:
    incarnation: int
    revision: int
    past: int
    total: int
    pages: tuple[Page, ...]
    slots: tuple[int, ...]
    external_writer_pending: bool = False


@dataclass(frozen=True)
class CpuUpload:
    past: tuple[int, ...]
    total: tuple[int, ...]
    offsets: tuple[int, ...]
    tables: tuple[tuple[int, ...], ...]
    slots: tuple[int, ...]
    cu: tuple[int, ...]


@dataclass(frozen=True)
class Snapshot:
    pool_epoch: int
    page_size: int
    page_count: int
    rows: tuple[OwnerRow, ...]
    upload: CpuUpload


def cpu_value(past, total, offsets, tables, slots, cu):
    """Copy ACTUAL CPU tensor values into deeply immutable values.

    Adapter must first check CPU/device, int32 lengths/tables/offsets, int64
    slots, ranks/contiguity and exact sizes. Never .to(cpu) device metadata.
    """
    return CpuUpload(tuple(past), tuple(total), tuple(offsets),
                     tuple(tuple(row) for row in tables), tuple(slots), tuple(cu))


def _validate(pool_epoch, page_size, page_count, rows, upload):
    integer(pool_epoch, 1, U64_MAX, "pool epoch")
    integer(page_size, 1, I32_MAX, "page size")
    integer(page_count, 1, I32_MAX, "page count")
    n = len(rows)
    if n == 0 or not (len(upload.past) == len(upload.total) == len(upload.tables) == n):
        raise ValueError("row count")
    if page_size % 64 or n > page_count or len(upload.offsets) != n + 1 or len(upload.cu) != n + 1:
        raise ValueError("metadata geometry")
    if upload.offsets[0] != 0 or upload.cu[0] != 0:
        raise ValueError("metadata initial offset")
    token_count = total_count = 0
    raw_writers = set()
    batch_pages = set()
    one_tail = all(r.total == r.past + 1 for r in rows)
    for offset in upload.offsets:
        integer(offset, 0, I32_MAX, "uploaded offset")
    for value in upload.cu:
        integer(value, 0, I32_MAX, "uploaded cu")
    for value in upload.past:
        integer(value, 0, I32_MAX - 1, "uploaded past")
    for value in upload.total:
        integer(value, 1, I32_MAX, "uploaded total")
    for value in upload.slots:
        integer(value, 0, (1 << 63) - 1, "uploaded slot")
    widths = {len(table) for table in upload.tables}
    if len(widths) != 1 or 0 in widths or max(widths) > page_count:
        raise ValueError("nonrectangular/empty upload table")
    seen = set()
    for i, row in enumerate(rows):
        integer(row.incarnation, 1, U64_MAX, "incarnation")
        integer(row.revision, 1, U64_MAX, "revision")
        if row.incarnation in seen or row.external_writer_pending:
            raise ValueError("duplicate owner or external writer")
        seen.add(row.incarnation)
        integer(row.past, 0, I32_MAX - 1, "past")
        integer(row.total, 1, I32_MAX, "total")
        if row.total <= row.past or row.total > page_count * page_size:
            raise ValueError("positive suffix required")
        if upload.past[i] != row.past or upload.total[i] != row.total:
            raise ValueError("submitted length differs from owner")
        count = (row.total - 1) // page_size + 1
        if len(row.pages) != count or len(upload.tables[i]) < count:
            raise ValueError("missing active pages")
        physical = set()
        for j, page in enumerate(row.pages):
            integer(page.physical, 0, page_count - 1, "physical page")
            integer(page.generation, 1, U64_MAX, "allocation generation")
            if page.physical in physical or upload.tables[i][j] != page.physical:
                raise ValueError("duplicate/mismatched active page")
            physical.add(page.physical)
            if one_tail and page.physical in batch_pages:
                raise ValueError("NSA initial decode domain excludes shared physical prefix pages")
            batch_pages.add(page.physical)
        for value in upload.tables[i]:
            integer(value, -1, page_count - 1, "uploaded table value")
        expected = tuple(row.pages[t // page_size].physical * page_size + t % page_size
                         for t in range(row.past, row.total))
        if row.slots != expected or upload.slots[token_count:token_count + len(expected)] != expected:
            raise ValueError("missing/fabricated/mismatched owner slot")
        if any(slot in raw_writers for slot in expected):
            raise ValueError("overlapping raw writers")
        raw_writers.update(expected)
        token_count += row.total - row.past
        total_count += row.total
        integer(token_count, 1, I32_MAX, "submitted token count")
        integer(total_count, 1, I32_MAX, "submitted total count")
        if upload.offsets[i + 1] != token_count or upload.cu[i + 1] != total_count:
            raise ValueError("offset/cu mismatch")
    if len(upload.slots) != token_count:
        raise ValueError("slot upload size mismatch")



def freeze_snapshot(pool_epoch, page_size, page_count, rows, upload):
    # Reconstruct immutable owner values: never retain mutable request lists.
    rows = tuple(OwnerRow(r.incarnation, r.revision, r.past, r.total,
                          tuple(Page(p.physical, p.generation) for p in r.pages),
                          tuple(r.slots), r.external_writer_pending) for r in rows)
    upload = cpu_value(upload.past, upload.total, upload.offsets,
                       upload.tables, upload.slots, upload.cu)
    _validate(pool_epoch, page_size, page_count, rows, upload)
    return Snapshot(pool_epoch, page_size, page_count, rows, upload)


def bind_effective_upload(snapshot, actual_local, actual_effective):
    """Worker preflight BEFORE compiler/to_model_input/cache write.

    Check exact original upload, then approved CPU bucket normalization. Padding
    and width may change; active pages/rows/slots/lengths must match. Root owns
    transformation and retention of the actual effective CPU tensor snapshot.
    """
    if actual_local != snapshot.upload:
        raise ValueError("submitted CPU metadata changed after freeze")
    _validate(snapshot.pool_epoch, snapshot.page_size, snapshot.page_count,
              snapshot.rows, actual_effective)
    return replace(snapshot, upload=cpu_value(
        actual_effective.past, actual_effective.total, actual_effective.offsets,
        actual_effective.tables, actual_effective.slots, actual_effective.cu))


@dataclass(frozen=True)
class OwnerBatch:
    pool_epoch: int
    page_size: int
    page_count: int
    rows: tuple[OwnerRow, ...]


def capture_owner_batch(manager, requests, is_prefill):
    clock = manager._nsa_owner_clock
    if clock is None:
        return None
    rows = tuple(OwnerRow(
        r._nsa_incarnation, r._nsa_prefix_revision, r.num_local_cached_tokens,
        r.get_prompt_length() if is_prefill else r.get_total_length(),
        tuple(Page(p, manager.blocks[p].allocation_generation) for p in r.block_table),
        tuple(r.slot_mapping)) for r in requests)
    one_tail = all(r.total == r.past + 1 for r in rows)
    manager.validate_nsa_request_writes(requests, decode=one_tail)
    return OwnerBatch(clock.pool_epoch, manager.block_size, manager.num_blocks, rows)


def freeze_model_input(engine, batch, model_input, core):
    """Actual processor CPU values; no device readback or request reconstruction."""
    if batch is None or engine.get_nsa_pool_epoch() != batch.pool_epoch:
        raise ValueError("NSA scheduler/worker pool binding absent or stale")
    if any(model_input.get(k) is not None for k in (
        "pixel_values", "image_bound", "tgt_sizes", "image_grid_thw",
        "image_req_ids", "visual_token_ranges", "target_hidden_states",
        "mamba_init_state_indices", "mamba_final_state_indices")):
        raise ValueError("NSA owner slice accepts ordinary text inputs only")
    if model_input.get("sample_all_positions", False):
        raise ValueError("NSA owner slice excludes speculative/all-position inputs")
    values = {}
    for key, dtype, rank in (
        ("past_kv_lengths", core.int32, 1), ("total_kv_lengths", core.int32, 1),
        ("input_offsets", core.int32, 1), ("cu_seqlens", core.int32, 1),
        ("block_tables", core.int32, 2), ("slot_mapping", core.int64, 1)):
        t = model_input.get(key)
        if (t is None or t.device.type != "cpu" or t.dtype != dtype
                or t.ndim != rank or not t.is_contiguous()):
            raise ValueError("NSA requires actual contiguous CPU metadata: " + key)
        values[key] = t.to_numpy().tolist()
    upload = cpu_value(values["past_kv_lengths"], values["total_kv_lengths"],
                       values["input_offsets"], values["block_tables"],
                       values["slot_mapping"], values["cu_seqlens"])
    s = freeze_snapshot(batch.pool_epoch, batch.page_size, batch.page_count, batch.rows, upload)
    rows = [(r.incarnation, r.revision, r.past, r.total,
             [(p.physical, p.generation) for p in r.pages], list(r.slots)) for r in s.rows]
    native = engine.NsaOwnerSnapshot(s.pool_epoch, s.page_size, s.page_count, rows,
        list(s.upload.past), list(s.upload.total), list(s.upload.offsets),
        list(s.upload.cu), [list(t) for t in s.upload.tables], list(s.upload.slots))
    # Private upload copies after validation; external CPU Tensor aliases are not
    # retained. The native worker repeats original/effective checks on its copies.
    for key, dtype in (("past_kv_lengths", core.int32), ("total_kv_lengths", core.int32),
                       ("input_offsets", core.int32), ("cu_seqlens", core.int32),
                       ("block_tables", core.int32), ("slot_mapping", core.int64)):
        model_input[key] = core.from_list(values[key], dtype=dtype)
    model_input["nsa_owner_snapshot"] = native
