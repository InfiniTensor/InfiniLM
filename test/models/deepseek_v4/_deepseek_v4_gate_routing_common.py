#!/usr/bin/env python3
"""Small DeepSeek V4 gate-routing fixtures shared by InfiniLM/SGLang tests.

The tests intentionally use tiny hand-written tensors.  They verify the two
gate routing modes without loading a production checkpoint:

* hash routing: selected experts come from ffn.gate.tid2eid (I64)
* noaux_tc routing: selected experts come from score + ffn.gate.bias (F32)

The gate projection weight is BF16, matching DeepSeek V4 checkpoints.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file


TOP_K = 2
EPS = 1e-9


@dataclass(frozen=True)
class GateFixture:
    hidden_states: torch.Tensor
    input_ids: torch.Tensor
    tid2eid: torch.Tensor
    weight: torch.Tensor
    bias: torch.Tensor
    scoring_func: str = "sigmoid"


def make_fixture() -> GateFixture:
    hidden_states = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.bfloat16,
    )
    weight = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.2, 0.0, 0.0],
            [0.5, 1.4, 0.2, 0.0],
            [-0.1, 0.1, 2.0, 0.0],
            [0.3, -0.5, 0.6, 0.0],
        ],
        dtype=torch.bfloat16,
    )
    bias = torch.tensor([0.0, -0.2, 0.4, 0.15, -0.05], dtype=torch.float32)
    tid2eid = torch.tensor(
        [
            [1, 2],
            [0, 3],
            [2, 4],
            [4, 0],
            [3, 1],
            [2, 3],
        ],
        dtype=torch.int64,
    )
    input_ids = torch.tensor([3, 0, 5], dtype=torch.int64)
    return GateFixture(
        hidden_states=hidden_states,
        input_ids=input_ids,
        tid2eid=tid2eid,
        weight=weight,
        bias=bias,
    )


def assert_fixture_dtypes(fixture: GateFixture) -> None:
    assert fixture.tid2eid.dtype == torch.int64, fixture.tid2eid.dtype
    assert fixture.weight.dtype == torch.bfloat16, fixture.weight.dtype
    assert fixture.bias.dtype == torch.float32, fixture.bias.dtype


def router_logits(fixture: GateFixture) -> torch.Tensor:
    # InfiniLM and the current SGLang reference both use FP32 accumulation for
    # the router projection while preserving BF16 gate weights on load.
    return fixture.hidden_states.float().matmul(fixture.weight.float().t())


def router_scores(logits: torch.Tensor, scoring_func: str) -> torch.Tensor:
    if scoring_func == "softmax":
        return F.softmax(logits, dim=-1)
    if scoring_func == "sigmoid":
        return torch.sigmoid(logits)
    if scoring_func == "sqrtsoftplus":
        return torch.sqrt(F.softplus(logits))
    raise ValueError(f"unsupported scoring_func={scoring_func!r}")


def normalize_selected_weights(
    scores: torch.Tensor,
    selected_ids: torch.Tensor,
    scoring_func: str,
) -> torch.Tensor:
    weights = scores.gather(dim=-1, index=selected_ids)
    if scoring_func != "softmax":
        weights = weights / (weights.sum(dim=-1, keepdim=True) + EPS)
    return weights


def hash_routing_reference(fixture: GateFixture) -> tuple[torch.Tensor, torch.Tensor]:
    assert_fixture_dtypes(fixture)
    logits = router_logits(fixture)
    scores = router_scores(logits, fixture.scoring_func)
    selected_ids = fixture.tid2eid[fixture.input_ids]
    selected_weights = normalize_selected_weights(
        scores, selected_ids, fixture.scoring_func
    )
    return selected_ids, selected_weights


def noaux_tc_routing_reference(fixture: GateFixture) -> tuple[torch.Tensor, torch.Tensor]:
    assert_fixture_dtypes(fixture)
    logits = router_logits(fixture)
    scores = router_scores(logits, fixture.scoring_func)
    biased_scores = scores + fixture.bias
    selected_rows = []
    for row in biased_scores:
        ranked = sorted(
            range(row.numel()),
            key=lambda expert_id: (-float(row[expert_id]), expert_id),
        )
        selected_rows.append(ranked[:TOP_K])
    selected_ids = torch.tensor(selected_rows, dtype=torch.int64)
    selected_weights = normalize_selected_weights(
        scores, selected_ids, fixture.scoring_func
    )
    return selected_ids, selected_weights


def assert_close(name: str, got: torch.Tensor, ref: torch.Tensor) -> None:
    if got.dtype.is_floating_point:
        if not torch.allclose(got.float(), ref.float(), atol=1e-6, rtol=1e-6):
            diff = (got.float() - ref.float()).abs()
            raise AssertionError(
                f"{name} mismatch: max_abs={float(diff.max())}, "
                f"mean_abs={float(diff.mean())}"
            )
        return
    if not torch.equal(got, ref):
        raise AssertionError(f"{name} mismatch: got={got.tolist()} ref={ref.tolist()}")


def expected_hash_ids() -> torch.Tensor:
    return torch.tensor([[4, 0], [1, 2], [2, 3]], dtype=torch.int64)


def expected_noaux_tc_ids() -> torch.Tensor:
    return torch.tensor([[2, 3], [2, 3], [3, 2]], dtype=torch.int64)


def make_hf_gate_state_dict(fixture: GateFixture) -> Dict[str, torch.Tensor]:
    return {
        "layers.0.ffn.gate.tid2eid": fixture.tid2eid,
        "layers.0.ffn.gate.weight": fixture.weight,
        "layers.3.ffn.gate.weight": fixture.weight.clone(),
        "layers.3.ffn.gate.bias": fixture.bias,
    }


EXPECTED_HF_DTYPES = {
    "layers.0.ffn.gate.tid2eid": torch.int64,
    "layers.0.ffn.gate.weight": torch.bfloat16,
    "layers.3.ffn.gate.weight": torch.bfloat16,
    "layers.3.ffn.gate.bias": torch.float32,
}


def assert_state_dict_dtype_and_value(
    loaded: Mapping[str, torch.Tensor],
    expected: Mapping[str, torch.Tensor],
) -> None:
    for name, ref in expected.items():
        if name not in loaded:
            raise AssertionError(f"missing loaded tensor: {name}")
        got = loaded[name]
        expected_dtype = EXPECTED_HF_DTYPES[name]
        if got.dtype != expected_dtype:
            raise AssertionError(
                f"{name} dtype mismatch: got={got.dtype}, expected={expected_dtype}"
            )
        assert_close(name, got, ref)


def roundtrip_hf_gate_state_dict() -> Dict[str, torch.Tensor]:
    fixture = make_fixture()
    expected = make_hf_gate_state_dict(fixture)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "tiny_deepseek_v4_gate.safetensors"
        save_file(expected, path)
        loaded = load_file(path)
    assert_state_dict_dtype_and_value(loaded, expected)
    return loaded


def _checkpoint_weight_map(checkpoint_dir: Path) -> Dict[str, str]:
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as f:
            return json.load(f)["weight_map"]
    weight_map: Dict[str, str] = {}
    for shard in sorted(checkpoint_dir.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as f:
            for key in f.keys():
                weight_map[key] = shard.name
    return weight_map


def load_checkpoint_tensor(checkpoint_dir: str | Path, key: str) -> torch.Tensor:
    checkpoint_dir = Path(checkpoint_dir)
    weight_map = _checkpoint_weight_map(checkpoint_dir)
    if key not in weight_map:
        raise KeyError(f"{key} not found in {checkpoint_dir}")
    shard = checkpoint_dir / weight_map[key]
    with safe_open(shard, framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def inspect_checkpoint_gate_dtypes(
    checkpoint_dir: str | Path,
    keys: Iterable[str] = EXPECTED_HF_DTYPES.keys(),
) -> Dict[str, torch.Tensor]:
    loaded: Dict[str, torch.Tensor] = {}
    for key in keys:
        tensor = load_checkpoint_tensor(checkpoint_dir, key)
        expected_dtype = EXPECTED_HF_DTYPES[key]
        if tensor.dtype != expected_dtype:
            raise AssertionError(
                f"{key} dtype mismatch: got={tensor.dtype}, expected={expected_dtype}"
            )
        loaded[key] = tensor
        sample = tensor.reshape(-1)[: min(6, tensor.numel())]
        if tensor.dtype == torch.bfloat16:
            sample = sample.float()
        print(
            f"{key}: dtype={tensor.dtype}, shape={tuple(tensor.shape)}, "
            f"sample={sample.tolist()}"
        )
    return loaded


def run_reference_gate_tests(label: str) -> None:
    fixture = make_fixture()
    assert_fixture_dtypes(fixture)

    hash_ids, hash_weights = hash_routing_reference(fixture)
    noaux_ids, noaux_weights = noaux_tc_routing_reference(fixture)

    assert_close(f"{label}.hash.ids", hash_ids, expected_hash_ids())
    assert_close(f"{label}.noaux_tc.ids", noaux_ids, expected_noaux_tc_ids())
    if hash_weights.dtype != torch.float32:
        raise AssertionError(f"{label}.hash.weights dtype={hash_weights.dtype}")
    if noaux_weights.dtype != torch.float32:
        raise AssertionError(f"{label}.noaux_tc.weights dtype={noaux_weights.dtype}")

    loaded = roundtrip_hf_gate_state_dict()
    print(f"{label}: hash ids={hash_ids.tolist()}")
    print(f"{label}: hash weights={hash_weights.tolist()}")
    print(f"{label}: noaux_tc ids={noaux_ids.tolist()}")
    print(f"{label}: noaux_tc weights={noaux_weights.tolist()}")
    print(f"{label}: safetensors roundtrip keys={sorted(loaded)}")
