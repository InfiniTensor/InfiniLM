import argparse, ctypes, hashlib, json, os, time, traceback
from pathlib import Path
import numpy as np
import threading

p = argparse.ArgumentParser()
p.add_argument("--mode", required=True)
p.add_argument("--scenario", default="perf")
p.add_argument("--out", required=True)
p.add_argument("--validation-root", default=os.environ.get("INFINILM_VALIDATION_ROOT"))
a = p.parse_args()
if not a.validation_root:
    p.error("provide --validation-root or INFINILM_VALIDATION_ROOT")
R = Path(a.validation_root)
result = dict(mode=a.mode, scenario=a.scenario, status="RUNNING", runs=[])


class Memory(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint),
        ("total", ctypes.c_ulonglong),
        ("reserved", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


nvml = ctypes.CDLL("libnvidia-ml.so.1")
assert nvml.nvmlInit_v2() == 0
handles = []
for u in os.environ["CUDA_VISIBLE_DEVICES"].split(","):
    h = ctypes.c_void_p()
    assert nvml.nvmlDeviceGetHandleByUUID(u.encode(), ctypes.byref(h)) == 0
    handles.append(h)


def memory():
    values = []
    for h in handles:
        m = Memory()
        m.version = ctypes.sizeof(Memory) | (2 << 24)
        assert nvml.nvmlDeviceGetMemoryInfo_v2(h, ctypes.byref(m)) == 0
        values.append(m.used)
    return values


stop = threading.Event()
samples = []


def observe():
    while not stop.is_set():
        samples.append(dict(time=time.time(), bytes=memory()))
        stop.wait(0.02)


observer = threading.Thread(target=observe)
observer.start()
try:
    from infinilm import _core as core
    from infinilm.config.engine_config import EngineConfig
    from infinilm.llm.model_runner.model_runner import ModelRunner
    from infinilm.cache.cache import PagedKVCacheConfig
    import infinilm.infer_engine as im

    read = im.read_hf_config

    def config(path):
        c = read(path)
        c.update(infinilm_prefill_chunk_size=4096, infinilm_tp_vocab_sharding=True)
        return c

    im.read_hf_config = config
    profiles = (
        None
        if a.mode in ["reference", "baseline", "upload"]
        else [(b, k) for b in [1, 8, 16] for k in [2048, 4096, 0]]
    )
    if a.mode == "limited":
        profiles = [(b, 2048) for b in [1, 8, 16]]
    if a.mode == "bounded":
        profiles = [(b, k) for b in [1, 8, 16] for k in [2048, 4096, 8192]]
    kwargs = dict(
        model_path=str(R / "models/Qwen__Qwen3-14B"),
        device="cuda",
        dtype="bfloat16",
        tensor_parallel_size=2,
        cache_type="paged",
        num_blocks=544,
        block_size=256,
        max_batch_size=16,
        enable_graph=True,
        attn_backend="paged-attn",
        enable_prefix_caching=False,
        weight_load_mode="sync",
    )
    if a.mode != "reference":
        kwargs.update(
            graph_capture_profiles=profiles,
            graph_metadata_upload=a.mode in ["upload", "combined"],
        )
    cfg = EngineConfig(**kwargs)
    start = time.perf_counter()
    runner = ModelRunner(cfg, initialize_processor=False)
    engine = runner.model_engine
    result.update(
        load_seconds=time.perf_counter() - start,
        loaded_memory=memory(),
        profiles=profiles,
    )

    def tensor(value, dtype):
        return core.from_numpy(np.asarray(value, dtype=dtype))

    def table(row):
        return np.arange(row * 33, (row + 1) * 33, dtype=np.int32)

    def prefill(batch, length):
        for row in range(batch):
            engine.forward_raw(
                tensor([[42] * length], np.int64),
                position_ids=tensor(np.arange(length), np.int64),
                past_kv_lengths=tensor([0], np.int32),
                total_kv_lengths=tensor([length], np.int32),
                input_offsets=tensor([0, length], np.int32),
                cu_seqlens=tensor([0, length], np.int32),
                block_tables=tensor([table(row)], np.int32),
                slot_mapping=tensor(np.arange(length) + row * 33 * 256, np.int64),
                sample_all_positions=False,
            )

    def decode(batch, length):
        begin = time.perf_counter()
        args = dict(
            position_ids=tensor([length] * batch, np.int64),
            past_kv_lengths=tensor([length] * batch, np.int32),
            total_kv_lengths=tensor([length + 1] * batch, np.int32),
            input_offsets=tensor(np.arange(batch + 1), np.int32),
            cu_seqlens=tensor(np.arange(batch + 1), np.int32),
            block_tables=tensor(
                np.stack([table(row) for row in range(batch)]), np.int32
            ),
            slot_mapping=tensor(
                [row * 33 * 256 + length for row in range(batch)], np.int64
            ),
            sample_all_positions=False,
        )
        ids = tensor([[42] * batch], np.int64)
        prepared = time.perf_counter()
        out = engine.forward_raw(ids, **args)
        done = time.perf_counter()
        return dict(
            cpu_prepare_ms=(prepared - begin) * 1000,
            decode_including_sampling_readback_ms=(done - prepared) * 1000,
            tokens=out["output_ids"].to_numpy().tolist(),
            execution=dict(out["execution_report"]),
            memory=memory(),
        )

    cases = (
        [(8, 2047)]
        if a.scenario == "perf"
        else [(b, l) for b in [1, 8, 16] for l in [2047, 4095, 8191]]
    )
    if a.scenario == "boundaries":
        cases = [(b, length) for b in [1, 8, 16] for length in [2047, 4095, 8191]]
    for batch, length in cases:
        prefill(batch, length)
        for step in range(7 if a.scenario == "perf" else 2):
            row = decode(batch, length)
            row.update(batch=batch, length=length, index=step, warmup=step < 2)
            result["runs"].append(row)
            expected = (
                "eager"
                if (
                    (a.mode == "limited" and length + 1 > 2048)
                    or (a.mode == "bounded" and length + 1 > 8192)
                )
                else "device_graph"
            )
            assert (
                row["execution"]["mode"] == expected
                and row["execution"]["host_segments"] == 0
            ), row
            if expected == "eager":
                assert (
                    row["execution"]["fallback_reason"] == "graph_profile_kv_bound_miss"
                ), row
        if a.scenario == "boundaries":
            # Previous decode wrote the next prefix token; cross into the next page/bound.
            transition = decode(batch, length + 1)
            transition.update(batch=batch, length=length + 1, boundary_transition=True)
            result.setdefault("boundary_transitions", []).append(transition)
            expected = (
                "eager"
                if (a.mode == "limited" or (a.mode == "bounded" and length + 2 > 8192))
                else "device_graph"
            )
            assert transition["execution"]["mode"] == expected, transition
        Path(a.out).with_suffix(".progress.json").write_text(json.dumps(result))
    if a.scenario != "perf":
        # Different batch and KV coverage misses must remain eager with explicit reasons.
        for b in [2]:
            prefill(b, 16)
            miss = decode(b, 16)
            result["batch_miss"] = miss
            if profiles is not None:
                assert (
                    miss["execution"]["fallback_reason"] == "graph_profile_batch_miss"
                ), miss
        engine.reset_cache(PagedKVCacheConfig(544, 256, 16))
        prefill(1, 16)
        result["after_reset"] = decode(1, 16)
        assert result["after_reset"]["execution"]["mode"] == "device_graph"
        if a.mode != "reference":
            invalid_table = np.stack([table(0), table(1)])
            invalid_table[0, 0] = -1
            try:
                engine.forward_raw(
                    tensor([[42, 42]], np.int64),
                    position_ids=tensor([16, 16], np.int64),
                    past_kv_lengths=tensor([16, 16], np.int32),
                    total_kv_lengths=tensor([17, 17], np.int32),
                    input_offsets=tensor([0, 1, 2], np.int32),
                    cu_seqlens=tensor([0, 1, 2], np.int32),
                    block_tables=tensor(invalid_table, np.int32),
                    slot_mapping=tensor([16, 33 * 256 + 16], np.int64),
                    sample_all_positions=False,
                )
                raise AssertionError("invalid table was not rejected")
            except (ValueError, RuntimeError) as error:
                assert "invalid paged graph page table" in str(error), str(error)
                result["invalid_table_rejected"] = str(error)
    site = os.environ["INFINILM_VALIDATION_SITE"]
    result["mapped_native"] = {
        f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
        for f in sorted(
            {
                l.split()[-1]
                for l in Path("/proc/self/maps").read_text().splitlines()
                if l.split()[-1].startswith(site) and ".so" in l.split()[-1]
            }
        )
    }
    result["status"] = "PASS"
    runner.close()
    engine = None
    import gc

    gc.collect()
except BaseException as e:
    result.update(status="FAIL", error=repr(e), traceback=traceback.format_exc())
    print(result["traceback"], flush=True)
finally:
    stop.set()
    observer.join()
    result["nvml_v2_samples"] = samples
    Path(a.out).write_text(json.dumps(result, indent=2))
raise SystemExit(result["status"] != "PASS")
