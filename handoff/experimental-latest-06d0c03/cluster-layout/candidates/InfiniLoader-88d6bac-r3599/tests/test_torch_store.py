import gc
from contextlib import contextmanager

import pytest

torch = pytest.importorskip("torch")

from test_shards import layout, write_tensor
from infiniloader.shards import build_plan, export_rank
from infiniloader.torch_store import TensorStore
from infiniloader.cache import load_model_into_memory, build_view


@contextmanager
def resident(path, view):
    model = load_model_into_memory(path, chunk_size=4096)
    try:
        build_view(model, view)
        yield view
    finally:
        model.close()


def test_cow_protection_and_outliving_store(tmp_path):
    source = tmp_path / "source"
    write_tensor(source)
    with resident(source, tmp_path / "view") as view:
        store = TensorStore(view)
        first = store.tensor("weight")
        other = TensorStore(view)
        untouched = other.tensor("weight")
    first.add_(10)
    assert untouched.flatten()[0].item() == 0
    tail = first[1:]
    del first
    store.close()
    gc.collect()
    assert tail.flatten()[0].item() == 22
    with pytest.raises(RuntimeError, match="closed"):
        store.tensor("weight")


def test_live_layout_switch_does_not_reopen_source(tmp_path, monkeypatch):
    source = tmp_path / "source"
    write_tensor(source)
    with resident(source, tmp_path / "view") as view:
        store = TensorStore(view)
        tensor = store.tensor("weight")
    # Once mapped, the original path is no longer needed for topology changes.
    (source / "weights.safetensors").unlink()
    ptr = tensor.untyped_storage().data_ptr()
    for tp in (4, 2, 1, 4):
        outputs = [store.rank_tensors(layout(0, tp), r)["weight"] for r in range(tp)]
        assert all(t.untyped_storage().data_ptr() == ptr for t in outputs)
        torch.testing.assert_close(torch.cat(outputs, dim=0), tensor)
    # Also test a strided inner-axis view without a hidden CPU contiguous copy.
    outputs = [store.rank_tensors(layout(1), r)["weight"] for r in range(2)]
    assert all(t.untyped_storage().data_ptr() == ptr for t in outputs)
    torch.testing.assert_close(torch.cat(outputs, dim=1), tensor)


def test_sharded_store_enforces_exported_contract(tmp_path):
    source = tmp_path / "source"
    write_tensor(source)
    rules = layout(1)
    plan = build_plan(source, rules)
    dest = tmp_path / "shard"
    export_rank(source, plan, 1, dest)
    with resident(dest, tmp_path / "shard-view") as view, TensorStore(view) as store:
        actual = store.rank_tensors(rules, 1)["weight"]
        with resident(source, tmp_path / "full-view") as full_view, TensorStore(full_view) as full:
            torch.testing.assert_close(actual, full.tensor("weight")[:, 3:, :])
        with pytest.raises(ValueError, match="exported"):
            store.rank_tensors(rules, 0)


def test_scalar_and_empty_cpu_view(tmp_path):
    scalar, empty = tmp_path / "scalar", tmp_path / "empty"
    write_tensor(scalar, shape=())
    write_tensor(empty, shape=(0, 2))
    with resident(scalar, tmp_path / "scalar-view") as view:
        assert TensorStore(view).tensor("weight").shape == ()
    with resident(empty, tmp_path / "empty-view") as view:
        assert TensorStore(view).tensor("weight").numel() == 0


def test_reject_mutable_source(tmp_path):
    source = tmp_path / "source"
    write_tensor(source)
    with pytest.raises(ValueError, match="sealed"):
        TensorStore(source)


def test_receipt_is_bound_to_mapped_weights(tmp_path, monkeypatch):
    import infiniloader.torch_store as module
    import struct
    source = tmp_path / "source"
    write_tensor(source)
    plan = build_plan(source, layout(0))
    first, second = tmp_path / "first", tmp_path / "second"
    export_rank(source, plan, 0, first)
    weights = source / "weights.safetensors"
    weights.write_bytes(weights.read_bytes()[:-4] + struct.pack("<f", 999))
    export_rank(source, build_plan(source, layout(0)), 1, second)
    real = module.verify_rank
    with resident(first, tmp_path / "view1") as a, resident(second, tmp_path / "view2") as b:
        def swap_after_verify(path):
            receipt = real(path)
            target = (b / "weights.safetensors").readlink()
            (a / "weights.safetensors").unlink()
            (a / "weights.safetensors").symlink_to(target)
            return receipt
        monkeypatch.setattr(module, "verify_rank", swap_after_verify)
        with pytest.raises(ValueError, match="receipt"):
            TensorStore(a)
