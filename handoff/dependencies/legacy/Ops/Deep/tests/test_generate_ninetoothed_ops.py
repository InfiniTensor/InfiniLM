import importlib.util
import pathlib
import sys
import tempfile
import types


def _load_generator_module(fake_ninetoothed, monkeypatch):
    path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "scripts"
        / "generate_ninetoothed_ops.py"
    )
    spec = importlib.util.spec_from_file_location(
        "ninetoothed_generator_under_test", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    monkeypatch.setitem(sys.modules, "ninetoothed", fake_ninetoothed)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


def test_generate_rms_norm_uses_ntops_premake_with_rank_configs(monkeypatch):
    calls = []

    fake_ninetoothed = types.SimpleNamespace(
        float32="ninetoothed.float32",
        float16="ninetoothed.float16",
        bfloat16="ninetoothed.bfloat16",
    )
    fake_ninetoothed.build = lambda *args, **kwargs: calls.append((args, kwargs))
    module = _load_generator_module(fake_ninetoothed, monkeypatch)

    fake_arrangement = object()
    fake_application = object()
    fake_tensors = object()
    premake_calls = []

    def fake_ntops_premake(*args, **kwargs):
        premake_calls.append((args, kwargs))
        return fake_arrangement, fake_application, fake_tensors

    fake_ntops = types.SimpleNamespace(
        kernels=types.SimpleNamespace(
            rms_norm=types.SimpleNamespace(premake=fake_ntops_premake)
        )
    )

    monkeypatch.setattr(module, "_build_manifest", lambda output_dir: ["kernel.cpp"])
    monkeypatch.setitem(sys.modules, "ntops", fake_ntops)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = pathlib.Path(tmpdir)
        manifest = module.generate(
            ["rms_norm"],
            output_dir=tmp_path,
        )

        assert manifest == ["kernel.cpp"]
        assert len(calls) == 1

        args, kwargs = calls[0]
        premake, configs = args
        assert premake is fake_ntops_premake
        assert configs == tuple(
            (
                (),
                {
                    "ndim": ndim,
                    "num_normalized_dims": 1,
                    "input_dtype": dtype,
                    "weight_dtype": dtype,
                    "output_dtype": dtype,
                    "block_size": block_size,
                },
                {},
            )
            for ndim in (2, 3, 4)
            for dtype in (
                "ninetoothed.float32",
                "ninetoothed.float16",
                "ninetoothed.bfloat16",
            )
            for block_size in (256, 512)
        )
        assert kwargs["caller"] == "cuda"
        assert kwargs["kernel_name"] == "infini_ops_ninetoothed_rms_norm"
        assert kwargs["output_dir"] == tmp_path / "rms_norm"
        assert kwargs["lazy"] is False
        assert kwargs["meta_parameters"] == ("block_size",)

        arrangement, application, tensors = fake_ntops_premake(
            ndim=2,
            num_normalized_dims=1,
            input_dtype="ninetoothed.float32",
            weight_dtype="ninetoothed.float32",
            output_dtype="ninetoothed.float32",
            block_size=256,
        )

        assert arrangement is fake_arrangement
        assert application is fake_application
        assert tensors is fake_tensors
        assert premake_calls == [
            (
                (),
                {
                    "ndim": 2,
                    "num_normalized_dims": 1,
                    "input_dtype": "ninetoothed.float32",
                    "weight_dtype": "ninetoothed.float32",
                    "output_dtype": "ninetoothed.float32",
                    "block_size": 256,
                },
            )
        ]
