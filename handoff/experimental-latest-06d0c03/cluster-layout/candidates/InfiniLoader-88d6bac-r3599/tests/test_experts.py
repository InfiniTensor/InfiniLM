from contextlib import contextmanager

import pytest

torch = pytest.importorskip("torch")
safe = pytest.importorskip("safetensors.torch")

from test_torch_store import resident
from infiniloader.torch_store import TensorStore
from infiniloader.experts import ExpertPool, hybrid_moe


@contextmanager
def moe(tmp_path, device=None, dtype=torch.float32):
    source = tmp_path / "source"
    source.mkdir()
    generator = torch.Generator().manual_seed(17)
    weights, groups = {}, {}
    for i in range(4):
        groups[str(i)] = {}
        for role, shape in {"up": (16, 8), "down": (8, 16), "up_bias": (16,), "down_bias": (8,)}.items():
            name = f"experts.{i}.{role}"
            groups[str(i)][role] = name
            weights[name] = (torch.randn(shape, generator=generator) * 0.1).to(dtype)
    safe.save_file(weights, str(source / "model.safetensors"))
    with resident(source, tmp_path / "view") as view, TensorStore(view) as store:
        one = sum(weights[n].numel() * weights[n].element_size() for n in groups["0"].values())
        with ExpertPool(store, groups, max_gpu_bytes=2 * one, device=device) as pool:
            yield pool, weights, groups, one


def reference(inputs, ids, gates, weights, groups):
    # Independent full CPU implementation, deliberately token/route based.
    output = torch.zeros_like(inputs)
    for token in range(len(inputs)):
        for slot in range(ids.shape[1]):
            p = {role: weights[name] for role, name in groups[str(ids[token, slot].item())].items()}
            hidden = (inputs[token] @ p["up"].T + p["up_bias"]).relu()
            output[token] += gates[token, slot] * (hidden @ p["down"].T + p["down_bias"])
    return output


def test_cpu_cold_experts_topk_repeats_and_empty_routes(tmp_path):
    with moe(tmp_path) as (pool, weights, groups, one):
        x = torch.arange(32, dtype=torch.float32).reshape(4, 8) / 32
        ids = torch.tensor([[0, 1], [2, 3], [1, 1], [3, 0]])
        gates = torch.tensor([[.2, .8], [.7, .3], [.5, .5], [.9, .1]])
        torch.testing.assert_close(hybrid_moe(pool, x, ids, gates), reference(x, ids, gates, weights, groups))
        assert pool.status()["cpu_hits"] == 4 and pool.status()["gpu_hits"] == 0
        assert not pool.promote("0")
        assert hybrid_moe(pool, x[:0], ids[:0], gates[:0]).shape == (0, 8)
        with pool.acquire("0") as lease:
            with pytest.raises(RuntimeError, match="active leases"):
                pool.close()
        assert not lease.weights


@pytest.mark.parametrize("dtype,gate_dtype", [(torch.float32, torch.float64), (torch.bfloat16, torch.float32)])
def test_mixed_router_precision(tmp_path, dtype, gate_dtype):
    with moe(tmp_path, dtype=dtype) as (pool, weights, groups, one):
        inputs = torch.ones((1, 8), dtype=dtype)
        ids = torch.tensor([[0, 1]])
        gates = torch.tensor([[.25, .75]], dtype=gate_dtype)
        actual = hybrid_moe(pool, inputs, ids, gates)
        assert actual.dtype == inputs.dtype
        torch.testing.assert_close(actual, reference(inputs, ids, gates, weights, groups), atol=2e-3, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA/ROCm device")
def test_gpu_hot_cpu_cold_lru_budget_and_lease(tmp_path):
    with moe(tmp_path, "cuda:0") as (pool, weights, groups, one):
        assert pool.promote("0") and pool.promote("1")
        with pytest.raises(ValueError, match="stream"):
            with pool.acquire("0", stream=object()):
                pass
        assert pool.status()["active_leases"] == 0
        assert pool.entries["0"].users == 0
        assert pool.status()["gpu_weight_bytes"] == 2 * one
        with pool.acquire("0") as lease:
            assert lease.device.type == "cuda"
            with pytest.raises(RuntimeError, match="leased"):
                pool.evict("0")
            assert pool.promote("2")  # evicts 1, keeps leased 0
            with pool.acquire("2"):
                assert not pool.promote("3")  # all capacity is currently leased
        x = torch.randn(5, 8)
        ids = torch.tensor([[0, 1], [2, 3], [1, 1], [3, 0], [2, 0]])
        gates = torch.softmax(torch.randn(5, 2), dim=1)
        torch.testing.assert_close(hybrid_moe(pool, x, ids, gates), reference(x, ids, gates, weights, groups), rtol=1e-5, atol=1e-6)
        assert pool.status()["cpu_hits"] >= 2 and pool.status()["gpu_hits"] >= 2
        assert pool.status()["gpu_weight_bytes"] <= pool.limit
        pool.limit = one - 1
        pool.evict("0")
        pool.evict("2")
        assert not pool.promote("3")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA/ROCm device")
def test_gpu_cross_stream_completion_and_failed_group_is_not_published(tmp_path, monkeypatch):
    with moe(tmp_path, "cuda:0") as (pool, weights, groups, one):
        pool.promote("0")
        stream = torch.cuda.Stream()
        done = torch.cuda.Event()
        with pool.acquire("0") as lease:
            lease.record_stream(stream)
            with torch.cuda.stream(stream):
                result = lease.weights["up"] @ torch.ones(8, device="cuda")
                done.record(stream)
        pool.evict("0")
        assert done.query()
        torch.testing.assert_close(result.cpu(), weights[groups["0"]["up"]] @ torch.ones(8))
        real = pool._cpu
        class Failure:
            def to(self, *args, **kwargs):
                raise RuntimeError("injected transfer failure")
        monkeypatch.setattr(pool, "_cpu", lambda expert: {**real(expert), "down": Failure()})
        with pytest.raises(RuntimeError, match="injected"):
            pool.promote("1")
        assert pool.status()["gpu_weight_bytes"] == 0
        assert "1" not in pool.status()["resident_experts"]
