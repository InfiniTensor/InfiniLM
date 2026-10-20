import argparse
import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
safe = pytest.importorskip("safetensors.torch")
pytest.importorskip("numpy")

script = Path(__file__).resolve().parents[1] / "examples" / "distributed_mlp.py"
spec = importlib.util.spec_from_file_location("distributed_mlp_example", script)
mlp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mlp)


@pytest.mark.parametrize("mode,tp,pp", [("sharded", 4, 1), ("sharded", 1, 4),
                                       ("sharded", 2, 2), ("replicated", 4, 1), ("replicated", 1, 1)])
def test_real_collectives_and_layout_switches(tmp_path, mode, tp, pp):
    args = argparse.Namespace(root=tmp_path / "fixture", mode=mode, tp=tp, pp=pp,
                              output=tmp_path / "result.json", backend="gloo", device="cpu", timeout=25)
    mlp.run_local(args)
    result = mlp.read_json(args.output)
    assert result["ok"] and result["max_abs_error"] < 1e-6
    assert all(not r["source_available"] for r in result["ranks"])
    assert sum(r["canonical_tensor_bytes"] for r in result["ranks"]) == result["plan"]["aggregate_tensor_bytes"]
    if mode == "replicated":
        assert all(len(r["cases"]) == len(mlp.switch_cases(tp * pp)) for r in result["ranks"])
    # Independent official parser accepts each exported safetensors checkpoint.
    for rank in range(tp * pp):
        official = safe.load_file(str(args.root / f"rank{rank}" / "weights.safetensors.offline"))
        parts = result["plan"]["ranks"][rank]["tensors"]
        assert set(official) == set(parts)
        for name, tensor in official.items():
            assert list(tensor.shape) == parts[name]["shape"]


def test_corrupt_rank_fails_and_all_workers_are_reaped(tmp_path, monkeypatch):
    original_prepare = mlp.prepare
    def corrupt(root, *args):
        plan = original_prepare(root, *args)
        file = root / "rank1" / "weights.safetensors"
        file.write_bytes(file.read_bytes()[:-1])
        return plan
    original_popen, children = mlp.subprocess.Popen, []
    def remember(*args, **kwargs):
        p = original_popen(*args, **kwargs)
        children.append(p)
        return p
    monkeypatch.setattr(mlp, "prepare", corrupt)
    monkeypatch.setattr(mlp.subprocess, "Popen", remember)
    args = argparse.Namespace(root=tmp_path / "fixture", mode="sharded", tp=2, pp=1,
                              output=tmp_path / "result.json", backend="gloo", device="cpu", timeout=3)
    with pytest.raises(RuntimeError, match="worker failed"):
        mlp.run_local(args)
    assert len(children) == 2 and all(p.poll() is not None for p in children)
    assert not args.output.exists()


def test_every_switch_needs_one_output_and_legal_small_world():
    assert mlp.switch_cases(1) == [(1, 1)]
    assert mlp.switch_cases(2) == [(2, 1), (1, 2), (2, 1)]
    assert all(pp <= mlp.LAYERS for tp, pp in mlp.switch_cases(8))
    output = mlp.reference().tolist()
    rows = [{"cases": [{"tp": 1, "pp": 1, "output": output},
                       {"tp": 1, "pp": 1, "output": None}]}]
    with pytest.raises(RuntimeError, match="exactly one"):
        mlp.validate_outputs(rows)


def test_startup_barrier_rejects_missing_and_mixed_versions(tmp_path):
    with pytest.raises(TimeoutError):
        mlp.publish_start(tmp_path, 2, "plan", .01)
    for rank in range(2):
        (tmp_path / f"ready{rank}.json").write_text(__import__("json").dumps({
            "rank": rank, "plan_id": "wrong" if rank else "plan", "nonce": str(rank)}))
    with pytest.raises(ValueError, match="mixed"):
        mlp.publish_start(tmp_path, 2, "plan", .1)
    assert not (tmp_path / "start.json").exists()
