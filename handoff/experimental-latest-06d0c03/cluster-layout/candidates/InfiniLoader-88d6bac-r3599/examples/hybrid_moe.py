"""Small real CPU/GPU MoE validation; no model download or global GPU changes."""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from infiniloader.cache import build_view, load_model_into_memory
from infiniloader.experts import ExpertPool, hybrid_moe
from infiniloader.torch_store import TensorStore


def run(device):
    import torch
    from safetensors.torch import save_file
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(23)
    groups, weights = {}, {}
    for expert in range(8):
        groups[str(expert)] = {}
        for role, shape in {"up": (128, 64), "down": (64, 128)}.items():
            name = f"experts.{expert}.{role}"
            groups[str(expert)][role] = name
            weights[name] = torch.randn(shape, generator=generator) * .05
    with tempfile.TemporaryDirectory(prefix="iloader-moe-") as directory:
        root = Path(directory)
        source = root / "source"
        source.mkdir()
        save_file(weights, str(source / "weights.safetensors"))
        del weights
        model = load_model_into_memory(source, chunk_size=1024**2)
        try:
            build_view(model, root / "view")
            with TensorStore(root / "view") as store:
                # Ensure every cold lookup also comes from the immutable RAM store.
                (source / "weights.safetensors").unlink()
                one = 2 * 128 * 64 * 4
                x = torch.randn(32, 64, generator=generator)
                ids = torch.arange(64).reshape(32, 2) % 8
                gates = torch.softmax(torch.randn(32, 2, generator=generator), dim=1)
                expected = torch.zeros_like(x)
                for token in range(len(x)):
                    for slot in range(2):
                        names = groups[str(ids[token, slot].item())]
                        h = (x[token] @ store.tensor(names["up"]).T).relu()
                        expected[token] += gates[token, slot] * (h @ store.tensor(names["down"]).T)
                if device is not None:
                    torch.cuda.set_device(torch.device(device))
                    torch.cuda.reset_peak_memory_stats()
                    baseline_allocated = torch.cuda.memory_allocated()
                with ExpertPool(store, groups, max_gpu_bytes=2 * one, device=device) as pool:
                    phases = []
                    for hot in (("0", "1"), ("2", "3"), ("6", "7"), ("0", "1")):
                        start = time.perf_counter()
                        admitted = [pool.promote(e) for e in hot]
                        output = hybrid_moe(pool, x, ids, gates)
                        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)
                        phases.append(dict(hot=list(hot), admitted=admitted,
                                           max_abs_error=(output - expected).abs().max().item(),
                                           seconds=time.perf_counter() - start, pool=pool.status()))
                    result = {"ok": True, "gpu_validated": device is not None,
                              "scope": "reference dense ReLU MoE; no KTransformers kernel integration",
                              "cpu_tensor_bytes": store.catalog["tensor_bytes"], "cpu_source_available": False,
                              "gpu_weight_budget_bytes": 2 * one, "phases": phases, "torch": torch.__version__}
                    if device is not None:
                        result["gpu"] = {"name": torch.cuda.get_device_name(),
                                         "runtime": "rocm" if torch.version.hip else "cuda",
                                         "allocated_baseline": baseline_allocated,
                                         "peak_allocated": torch.cuda.max_memory_allocated(),
                                         "peak_reserved": torch.cuda.max_memory_reserved()}
                    return result
        finally:
            model.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu, cuda:0, ... (PyTorch ROCm also uses cuda)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(None if args.device == "cpu" else args.device)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
