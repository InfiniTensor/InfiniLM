"""Numerical TP/PP validation: local processes or one worker per SSH host.

Uses real torch.distributed collectives and activation transfers, not IO simulation.
Run `python examples/distributed_mlp.py --help`. Only generated fixtures are used.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

from infiniloader.cache import build_view, load_model_into_memory
from infiniloader.shards import build_plan, canonical, digest, export_rank, read_json
from infiniloader.torch_store import TensorStore
from infiniloader.runtime import atomic_json, process_token

DIM, HIDDEN, LAYERS = 32, 64, 4


def fixture():
    import torch
    g = torch.Generator().manual_seed(20261018)
    weights = {}
    for layer in range(LAYERS):
        for role, shape in {"up": (HIDDEN, DIM), "down": (DIM, HIDDEN),
                            "up_bias": (HIDDEN,), "down_bias": (DIM,)}.items():
            weights[f"layers.{layer}.{role}"] = torch.randn(shape, generator=g) * .05
    return weights


def layout(tp, pp):
    if LAYERS % pp:
        raise ValueError("fixture layers must be divisible by pp")
    return {"format": 1, "tp": tp, "pp": pp, "tensors": {
        f"layers.{layer}.{role}": {"stage": layer // (LAYERS // pp), "split_dim": dim}
        for layer in range(LAYERS) for role, dim in {"up": 0, "down": 1, "up_bias": 0, "down_bias": None}.items()}}


def input_tensor():
    import torch
    return torch.arange(8 * DIM, dtype=torch.float32).reshape(8, DIM) / 256


def reference():
    import torch
    weights, x = fixture(), input_tensor()
    for layer in range(LAYERS):
        p = {r: weights[f"layers.{layer}.{r}"] for r in ("up", "down", "up_bias", "down_bias")}
        x = torch.nn.functional.linear(torch.nn.functional.linear(x, p["up"], p["up_bias"]).relu(), p["down"], p["down_bias"])
    return x


def switch_cases(world):
    cases = [(world // pp, pp) for pp in (1, 2, 4) if world % pp == 0 and pp <= LAYERS]
    return cases + cases[:1] if len(cases) > 1 else cases


def validate_outputs(rows):
    import torch
    if not rows or not rows[0]["cases"] or any(len(r["cases"]) != len(rows[0]["cases"]) for r in rows):
        raise RuntimeError("missing topology results")
    errors, expected = [], reference()
    for index in range(len(rows[0]["cases"])):
        cases = [r["cases"][index] for r in rows]
        if len({(c["tp"], c["pp"]) for c in cases}) != 1:
            raise RuntimeError("ranks disagree on topology")
        outputs = [c["output"] for c in cases if c["output"] is not None]
        if len(outputs) != 1:
            raise RuntimeError("each topology must produce exactly one terminal output")
        actual = torch.tensor(outputs[0])
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        errors.append((actual - expected).abs().max().item())
    return errors


def prepare(root: Path, mode, tp, pp):
    from safetensors.torch import save_file
    root.mkdir(parents=True, exist_ok=False)
    full = root / "canonical"
    full.mkdir()
    save_file(fixture(), str(full / "weights.safetensors"))
    plan = build_plan(full, layout(tp, pp), mode=mode)
    for rank in range(tp * pp):
        export_rank(full, plan, rank, root / f"rank{rank}")
    (root / "fixture.json").write_bytes(canonical({"format": 1, "kind": "generated-mlp", "plan": plan}))
    return plan


def memory():
    result = {}
    for line in Path("/proc/self/smaps_rollup").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in ("Rss", "Pss", "Private_Dirty", "Private_Clean", "Shared_Dirty", "Shared_Clean"):
            result[key + "_bytes"] = int(value.split()[0]) * 1024
    return result


def publish_start(directory: Path, world: int, plan_id: str, timeout: float):
    """Controller-side release after every independently launched rank is ready."""
    deadline = time.monotonic() + timeout
    paths = [directory / f"ready{r}.json" for r in range(world)]
    while not all(p.exists() for p in paths):
        if time.monotonic() >= deadline:
            raise TimeoutError("not every rank reached the startup barrier")
        time.sleep(.05)
    rows = [read_json(p) for p in paths]
    if any(r.get("rank") != i or r.get("plan_id") != plan_id or not isinstance(r.get("nonce"), str)
           for i, r in enumerate(rows)) or len({r["nonce"] for r in rows}) != world:
        raise ValueError("mixed versions/ranks at startup barrier")
    marker = directory / "start.json"
    if marker.exists():
        raise FileExistsError("startup barrier was already released")
    atomic_json(marker, {"format": 1, "plan_id": plan_id,
                         "participants": {str(r["rank"]): r["nonce"] for r in rows}})
    return rows


def wait_start(directory: Path, rank: int, world: int, plan_id: str, timeout: float):
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("startup barrier requires a fresh shared directory")
    ready = {"rank": rank, "plan_id": plan_id, "nonce": uuid.uuid4().hex,
             "hostname": socket.gethostname(), "pid": os.getpid(), "process_token": process_token(os.getpid())}
    path = directory / f"ready{rank}.json"
    # Each operation uses a fresh barrier; never join stale readiness records.
    if path.exists():
        raise FileExistsError(path)
    atomic_json(path, ready)
    deadline = time.monotonic() + timeout
    while True:
        marker = directory / "start.json"
        if marker.exists():
            start = read_json(marker)
            if (start.get("format") != 1 or start.get("plan_id") != plan_id
                    or set(start.get("participants", {})) != {str(r) for r in range(world)}
                    or start["participants"].get(str(rank)) != ready["nonce"]):
                raise ValueError("stale or mismatched startup release")
            return ready
        if time.monotonic() >= deadline:
            raise TimeoutError("controller did not release startup barrier")
        time.sleep(.05)


def execute(store, rule, rank, device):
    import torch
    import torch.distributed as dist
    tp, pp = rule["tp"], rule["pp"]
    stage, lane = divmod(rank, tp)
    groups = [dist.new_group(list(range(s * tp, (s + 1) * tp))) for s in range(pp)]
    try:
        # .to('cpu') preserves the CPU COW views; CUDA makes only this rank's copies.
        tensors = {n: t.to(device) for n, t in store.rank_tensors(rule, rank).items()}
        x = input_tensor().to(device) if stage == 0 else torch.empty((8, DIM), device=device)
        if stage:
            dist.recv(x, src=rank - tp)
        for layer in range(stage * (LAYERS // pp), (stage + 1) * (LAYERS // pp)):
            p = {r: tensors[f"layers.{layer}.{r}"] for r in ("up", "down", "up_bias", "down_bias")}
            hidden = torch.nn.functional.linear(x, p["up"], p["up_bias"]).relu()
            x = torch.nn.functional.linear(hidden, p["down"])
            if tp > 1:
                dist.all_reduce(x, group=groups[stage])
            x += p["down_bias"]  # Add replicated row-parallel bias exactly once.
        if stage < pp - 1:
            dist.send(x, dst=rank + tp)
        return x.cpu().tolist() if stage == pp - 1 and lane == 0 else None
    finally:
        for group in groups:
            if group != dist.GroupMember.NON_GROUP_MEMBER:
                dist.destroy_process_group(group)


def worker(args):
    import torch
    import torch.distributed as dist
    torch.set_num_threads(1)
    if args.device != "cpu":
        torch.cuda.set_device(torch.device(args.device))
        # Finish device runtime startup before reporting readiness to the controller.
        warmup = torch.zeros(1, device=args.device)
        torch.cuda.synchronize()
        del warmup
    meta = read_json(args.root / "fixture.json")
    if meta.get("kind") != "generated-mlp":
        raise ValueError("only generated fixture directories may be used")
    plan = meta["plan"]
    source = args.root / f"rank{args.rank}"
    if (source / "weights.safetensors.offline").exists():
        raise ValueError("fixture already consumed; prepare a fresh one")
    with tempfile.TemporaryDirectory(prefix="iloader-mlp-") as temp:
        model, store = None, None
        try:
            start = time.perf_counter()
            model = load_model_into_memory(source, chunk_size=1024 * 1024, max_memory_bytes=256 * 1024**2)
            view = Path(temp) / "view"
            build_view(model, view)
            store = TensorStore(view)
            if store.content_id != plan["content_id"]:
                raise ValueError("fixture content mismatch")
            # Remove original fixture access. All later TP/PP switches must use RAM.
            (source / "weights.safetensors").rename(source / "weights.safetensors.offline")
            info = {"rank": args.rank, "hostname": socket.gethostname(),
                    "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                    "content_id": store.content_id, "plan_id": plan["plan_id"],
                    "backend": args.backend, "device": args.device, "torch": torch.__version__,
                    "canonical_tensor_bytes": store.catalog["tensor_bytes"],
                    "resident_file_bytes": model.total_bytes, "preload_seconds": time.perf_counter() - start,
                    "memory_after_map": memory(), "source_available": False}
            if args.ready_dir:
                info["startup"] = wait_start(args.ready_dir, args.rank, args.world_size, plan["plan_id"], args.timeout)
            dist.init_process_group(args.backend, init_method=args.init, rank=args.rank,
                                    world_size=args.world_size, timeout=datetime.timedelta(seconds=args.timeout))
            identities = [None] * args.world_size
            dist.all_gather_object(identities, (store.content_id, plan["plan_id"]))
            if len(set(identities)) != 1 or args.world_size != len(plan["ranks"]):
                raise ValueError("mixed checkpoint/layout versions or world size")
            cases = [(plan["layout"]["tp"], plan["layout"]["pp"])]
            if plan["mode"] == "replicated" and args.switch:
                cases = switch_cases(args.world_size)
            results = []
            for tp, pp in cases:
                before = time.perf_counter()
                output = execute(store, layout(tp, pp), args.rank, args.device)
                dist.barrier()
                if args.device != "cpu":
                    torch.cuda.synchronize()
                results.append({"tp": tp, "pp": pp, "seconds": time.perf_counter() - before,
                                "output": output, "memory": memory()})
            info["cases"] = results
            info["ok"] = True
            args.output.write_text(json.dumps(info, indent=2))
        finally:
            if store is not None:
                store.close()
            if model is not None:
                model.close()
            if dist.is_initialized():
                dist.destroy_process_group()


def run_local(args):
    import torch
    root = args.root
    plan = prepare(root, args.mode, args.tp, args.pp)
    world = args.tp * args.pp
    barrier = root / "startup"
    barrier.mkdir()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    processes, logs = [], []
    start = time.perf_counter()
    try:
        for rank in range(world):
            output = root / f"result{rank}.json"
            log = (root / f"worker{rank}.log").open("w")
            logs.append(log)
            cmd = [sys.executable, str(Path(__file__).resolve()), "worker", "--root", str(root),
                   "--rank", str(rank), "--world-size", str(world), "--init", f"tcp://127.0.0.1:{port}",
                   "--backend", args.backend, "--device", args.device, "--output", str(output),
                   "--timeout", str(args.timeout)]
            cmd.extend(["--ready-dir", str(barrier)])
            if args.mode == "replicated":
                cmd.append("--switch")
            processes.append(subprocess.Popen(cmd, stdout=log, stderr=log))
        try:
            publish_start(barrier, world, plan["plan_id"], args.timeout)
        except TimeoutError:
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError(f"worker failed before startup; inspect {root}/worker*.log") from None
            raise
        deadline = time.monotonic() + args.timeout + 20
        for p in processes:
            p.wait(timeout=max(.1, deadline - time.monotonic()))
        if any(p.returncode for p in processes):
            raise RuntimeError(f"worker failed; inspect {root}/worker*.log")
        rows = [read_json(root / f"result{r}.json") for r in range(world)]
        errors = validate_outputs(rows)
        result = {"ok": True, "scope": "single-host torch.distributed", "mode": args.mode,
                  "world_size": world, "plan": plan, "max_abs_error": max(errors), "ranks": rows,
                  "wall_seconds": time.perf_counter() - start}
        args.output.write_text(json.dumps(result, indent=2))
        print(json.dumps({k: v for k, v in result.items() if k not in ("ranks", "plan")}, indent=2))
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for log in logs:
            log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("prepare", "local", "worker"):
        p = sub.add_parser(action)
        p.add_argument("--root", type=Path, required=True)
        if action != "worker":
            p.add_argument("--mode", choices=("sharded", "replicated"), default="sharded")
            p.add_argument("--tp", type=int, default=4)
            p.add_argument("--pp", type=int, default=1)
        if action != "prepare":
            p.add_argument("--output", type=Path, required=True)
            p.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
            p.add_argument("--device", default="cpu")
            p.add_argument("--timeout", type=int, default=40)
        if action == "worker":
            p.add_argument("--rank", type=int, required=True)
            p.add_argument("--world-size", type=int, required=True)
            p.add_argument("--init", required=True)
            p.add_argument("--switch", action="store_true")
            p.add_argument("--ready-dir", type=Path, help="wait for controller release after all ranks preload")
    args = parser.parse_args()
    if args.action == "prepare":
        print(json.dumps(prepare(args.root, args.mode, args.tp, args.pp), indent=2))
    elif args.action == "worker":
        worker(args)
    else:
        run_local(args)


if __name__ == "__main__":
    main()
