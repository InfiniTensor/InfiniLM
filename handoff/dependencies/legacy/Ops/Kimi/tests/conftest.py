import hashlib
import random
import time

import pytest
import torch
import torch.utils.benchmark as benchmark

from tests import host_range_profile
from tests.report import register_reporter
from tests.utils import clone_strided, get_available_devices


def pytest_addoption(parser):
    parser.addoption(
        "--benchmark", action="store_true", help="Run performance benchmarks."
    )
    parser.addoption(
        "--host-range-profile",
        action="store",
        default=None,
        help="Write host-range benchmark timings as JSON lines.",
    )
    parser.addoption(
        "--devices",
        nargs="+",
        default=None,
        help="Device(s) to test on (e.g., `--devices ascend cpu`). Accepts platform names (`nvidia`, `mars`, `metax`, `iluvatar`, `hygon`, `thead`, `moore`, `cambricon`, `ascend`) or PyTorch device types (`cuda`, `mlu`, `musa`, `npu`). Defaults to all available devices.",
    )
    parser.addoption(
        "--report",
        action="store",
        default=None,
        help=(
            "Write a structured coverage report to the given JSON path. "
            "Also emits sibling `.details.jsonl` and `.summary.txt` files."
        ),
    )


def _host_range_profile_compiled():
    import infini.ops as ops

    compiled_fn = getattr(ops, "_host_range_profile_compiled", None)

    return callable(compiled_fn) and compiled_fn() is True


def pytest_configure(config):
    torch.backends.fp32_precision = "tf32"

    profile_output = config.getoption("--host-range-profile")
    config._infini_host_range_profile_output = None

    if profile_output:
        try:
            host_range_profile.validate_request(
                profile_output,
                benchmark_enabled=config.getoption("--benchmark"),
                compiled=_host_range_profile_compiled,
                xdist_workers=getattr(config.option, "numprocesses", None),
            )
        except ValueError as error:
            raise pytest.UsageError(str(error)) from error
        except Exception as error:
            raise pytest.UsageError(
                "host-range profiling is not compiled into infini.ops"
            ) from error

        config._infini_host_range_profile_output = host_range_profile.truncate_output(
            profile_output
        )

    config.addinivalue_line(
        "markers",
        "auto_act_and_assert: automatically perform Act and Assert phases using the return values",
    )
    register_reporter(config)


def pytest_collectstart(collector):
    if isinstance(collector, pytest.Module):
        _set_random_seed(_hash(collector.name))


@pytest.fixture(scope="module", autouse=True)
def set_seed_per_module(request):
    _set_random_seed(_hash(_module_path_from_request(request)))


@pytest.fixture(autouse=True)
def set_seed_per_test(request):
    _set_random_seed(_hash(_test_case_path_from_request(request)))


@pytest.fixture(scope="module", autouse=True)
def _clear_operator_caches():
    """Clear the C++ operator cache between test modules.

    The `Operator::call()` cache keys on tensor geometry (shape, strides,
    dtype) but not data pointers. When different test modules create tensors
    with identical geometry but different data content (e.g., random
    `cos_sin_cache` tables), a stale cached operator from a prior module
    silently returns wrong results. Clearing the cache at module boundaries
    ensures each module starts with a cold cache.
    """
    yield

    try:
        import infini.ops as ops

        for name in dir(ops):
            cls = getattr(ops, name)

            if hasattr(cls, "clear_cache"):
                cls.clear_cache()
    except ImportError:
        pass


_NPU_UNSUPPORTED_DTYPES = {torch.float64}

# `torch_npu` does not implement random number generation for
# `uint16`/`uint32`/`uint64`.
for _bits in (16, 32, 64):
    _t = getattr(torch, f"uint{_bits}", None)

    if _t is not None:
        _NPU_UNSUPPORTED_DTYPES.add(_t)


@pytest.fixture(autouse=True)
def skip_unsupported_dtypes(request):
    if not hasattr(request.node, "callspec"):
        return

    params = request.node.callspec.params

    if params.get("device") == "npu" and params.get("dtype") in _NPU_UNSUPPORTED_DTYPES:
        pytest.skip(f"{params['dtype']} not supported on Ascend 910B")


@pytest.fixture(autouse=True)
def skip_op_without_platform_impl(request):
    """Skip `device=<torch_type>` parametrizations when the op has no
    implementation on any of the corresponding platforms.

    Only runs for tests that parametrize `device` but not
    `implementation_index` — joint `(device, impl_idx)` parametrize in
    `pytest_generate_tests` already prunes empty-impl pairs at collection
    time, making this check redundant (and wasteful) on those tests.
    """

    if not hasattr(request.node, "callspec"):
        return

    params = request.node.callspec.params

    if "implementation_index" in params:
        return

    device_selectors = _active_device_selectors_for_torch_device(
        request.config, params.get("device")
    )

    if not device_selectors:
        return

    op_cls = _op_class_from_module(request.node.module)

    if op_cls is None:
        if "op_meta" in params:
            return

        pytest.skip("operator wrapper is not available in this build")

    if not hasattr(op_cls, "active_implementation_indices"):
        return

    if not any(op_cls.active_implementation_indices(d) for d in device_selectors):
        pytest.skip(
            f"{op_cls.__name__} has no implementation on any "
            f"`{params.get('device')}`-mapped platform/device"
        )


def _set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


_PLATFORM_TO_TORCH_DEVICE = {
    "nvidia": "cuda",
    "mars": "cuda",
    "metax": "cuda",
    "iluvatar": "cuda",
    "hygon": "cuda",
    "thead": "cuda",
    "moore": "musa",
    "cambricon": "mlu",
    "ascend": "npu",
}


_SMOKE_STATIC_TEST_MODULES = {
    "tests/test_cpp_api.py",
    "tests/test_generate_ninetoothed_ops.py",
    "tests/test_generate_torch_ops.py",
    "tests/test_generate_wrappers.py",
}

_SMOKE_TORCH_OPS = {"abs", "clamp", "exp", "relu"}


def pytest_itemcollected(item):
    if _is_smoke_item(item):
        item.add_marker("smoke")


def _is_smoke_item(item):
    module_path = _module_path_from_item(item)

    if module_path in _SMOKE_STATIC_TEST_MODULES:
        return True

    params = _callspec_params(item)

    if not params:
        return False

    matchers = {
        "tests/test_add.py": _is_smoke_add_case,
        "tests/test_cast.py": _is_smoke_cast_case,
        "tests/test_cat.py": _is_smoke_cat_case,
        "tests/test_causal_softmax.py": _is_smoke_causal_softmax_case,
        "tests/test_causal_softmax_infinilm.py": _is_smoke_causal_softmax_case,
        "tests/test_gemm.py": _is_smoke_gemm_case,
        "tests/test_linear.py": _is_smoke_linear_case,
        "tests/test_matmul.py": _is_smoke_matmul_case,
        "tests/test_moe_sum.py": _is_smoke_moe_sum_case,
        "tests/test_mul.py": _is_smoke_mul_case,
        "tests/test_relu.py": _is_smoke_relu_case,
        "tests/test_rms_norm.py": _is_smoke_rms_norm_case,
        "tests/test_silu_and_mul.py": _is_smoke_silu_and_mul_case,
        "tests/test_cutlass_scaled_mm.py": _is_smoke_cutlass_scaled_mm_case,
        "tests/test_flash_attn_varlen_func.py": _is_smoke_flash_attn_varlen_func_case,
        "tests/test_swiglu.py": _is_smoke_swiglu_case,
        "tests/test_torch_ops.py": _is_smoke_torch_op_case,
    }
    matcher = matchers.get(module_path)

    return matcher(params) if matcher else False


def _callspec_params(item):
    callspec = getattr(item, "callspec", None)

    return getattr(callspec, "params", {})


def _module_path_from_item(item):
    module = getattr(item, "module", None)

    if module is None:
        return ""

    return f"{module.__name__.replace('.', '/')}.py"


def _shape_case(params, *names):
    return tuple(params.get(name) for name in names)


def _is_float32(params, name="dtype"):
    return params.get(name) == torch.float32


def _is_smoke_add_case(params):
    cases = {
        ((13, 4), None, None, None),
        ((13, 4), (10, 1), (10, 1), (10, 1)),
        ((13, 4), (0, 1), None, None),
    }

    return (
        _is_float32(params)
        and params.get("alpha") is None
        and params.get("input_shape")
        == params.get("other_shape")
        == params.get("out_shape")
        and _shape_case(
            params,
            "input_shape",
            "input_strides",
            "other_strides",
            "out_strides",
        )
        in cases
    )


def _is_smoke_mul_case(params):
    cases = {
        ((13, 4), None, None, None),
        ((13, 4), (10, 1), (10, 1), (10, 1)),
    }

    if "input_shape" in params:
        return _is_float32(params) and (
            params.get("input_shape"),
            params.get("other_shape"),
            params.get("out_shape"),
        ) == (
            (1, 151936),
            (1, 1),
            (1, 151936),
        )

    return (
        _is_float32(params)
        and _shape_case(
            params,
            "shape",
            "input_strides",
            "other_strides",
            "out_strides",
        )
        in cases
    )


def _is_smoke_relu_case(params):
    if not _is_float32(params):
        return False

    if params.get("size") == 33:
        return True

    return _shape_case(
        params,
        "shape",
        "input_strides",
        "out_strides",
        "inplace",
    ) in {
        ((1, 3), None, None, False),
        ((1, 3), None, None, True),
    }


def _is_smoke_cast_case(params):
    cases = {
        ((13, 4), None, None, torch.float16, torch.float32),
        ((13, 4), (10, 1), (10, 1), torch.float32, torch.float16),
    }

    return (
        _shape_case(
            params,
            "shape",
            "input_strides",
            "out_strides",
            "input_dtype",
            "out_dtype",
        )
        in cases
    )


def _is_smoke_cat_case(params):
    cases = {
        (((4, 64), (4, 64)), 0, (8, 64)),
        (((4, 32), (4, 64)), -1, (4, 96)),
    }

    return (
        _is_float32(params)
        and _shape_case(
            params,
            "shapes",
            "dim",
            "out_shape",
        )
        in cases
    )


def _is_smoke_gemm_case(params):
    cases = {
        ((1, 2048), (2048, 2048), (1, 2048), None, None, None, 1, 1),
        ((4, 48, 64), (4, 64, 6), (4, 48, 6), None, None, None, 1, 0),
    }

    return (
        _is_float32(params)
        and params.get("trans_a") is False
        and params.get("trans_b") is False
        and _shape_case(
            params,
            "a_shape",
            "b_shape",
            "y_shape",
            "a_strides",
            "b_strides",
            "y_strides",
            "alpha",
            "beta",
        )
        in cases
    )


def _is_smoke_matmul_case(params):
    cases = {
        ((4, 64), (64, 32), (4, 32)),
        ((2, 4, 64), (2, 64, 32), (2, 4, 32)),
    }

    return _is_float32(params) and (
        _shape_case(params, "a_shape", "b_shape", "c_shape") in cases
    )


def _is_smoke_linear_case(params):
    cases = {
        (((4, 64), (32, 64), (4, 32)), False),
        (((2, 4, 64), (32, 64), (2, 4, 32)), True),
    }

    return (
        _is_float32(params)
        and (
            _shape_case(params, "input_shape", "weight_shape", "out_shape"),
            params.get("has_bias"),
        )
        in cases
    )


def _is_smoke_rms_norm_case(params):
    cases = {
        ((1, 64), (64,), None, None, None),
        ((2, 4, 2048), (2048,), None, None, None),
    }

    return (
        _is_float32(params)
        and params.get("eps") == 1e-6
        and _shape_case(
            params,
            "input_shape",
            "weight_shape",
            "input_strides",
            "weight_strides",
            "out_strides",
        )
        in cases
    )


def _is_smoke_moe_sum_case(params):
    return (
        params.get("topk") == 2
        and params.get("hidden_size") == 64
        and params.get("dtype") == torch.float16
    )


def _is_smoke_silu_and_mul_case(params):
    cases = {
        ((13, 4), None, None),
        ((16, 5632), None, None),
    }

    return (
        _is_float32(params)
        and _shape_case(
            params,
            "out_shape",
            "input_strides",
            "out_strides",
        )
        in cases
    )


def _is_smoke_cutlass_scaled_mm_case(params):
    return (
        params.get("m") == 16
        and params.get("n") == 32
        and params.get("k") == 64
        and params.get("per_token") is True
        and params.get("per_channel") is True
        and params.get("has_bias") is True
        and params.get("padded") is False
        and params.get("dtype") == torch.float16
    )


def _is_smoke_flash_attn_varlen_func_case(params):
    if params.get("device") == "cuda" and params.get("implementation_index") == 0:
        return (
            (
                params.get("q_lens"),
                params.get("k_lens"),
                params.get("num_heads"),
                params.get("head_dim"),
                params.get("use_alibi"),
            )
            in (
                ((2, 3), (130, 300), 4, 64, True),
                ((2, 3), (130, 300), 4, 128, True),
                ((13,), (13,), 32, 128, False),
            )
            and params.get("num_kv_heads") == 2
            and params.get("causal") is True
            and params.get("window_size") == (-1, -1)
            and params.get("scale") is None
            and params.get("paged") is True
            and params.get("dtype") == torch.float16
        )

    dense = (
        params.get("q_lens") == (3, 5)
        and params.get("k_lens") == (4, 5)
        and params.get("num_heads") == 4
        and params.get("num_kv_heads") == 4
        and params.get("causal") is False
        and params.get("paged") is False
    )
    paged = (
        params.get("q_lens") == (2, 3)
        and params.get("k_lens") == (130, 300)
        and params.get("num_heads") == 4
        and params.get("num_kv_heads") == 2
        and params.get("causal") is True
        and params.get("paged") is True
    )
    moore_paged_regression = (
        "q_lens" not in params
        and params.get("device") == "musa"
        and params.get("implementation_index") == 8
        and params.get("head_dim") == 64
    )
    return moore_paged_regression or (
        (dense or paged)
        and params.get("window_size") == (-1, -1)
        and params.get("scale") is None
        and params.get("head_dim") == 64
        and params.get("dtype") == torch.float16
    )


def _is_smoke_swiglu_case(params):
    cases = {
        ((13, 4), None, None, None),
        ((16, 5632), None, None, None),
    }

    return (
        _is_float32(params)
        and _shape_case(
            params,
            "shape",
            "input_strides",
            "gate_strides",
            "out_strides",
        )
        in cases
    )


def _is_smoke_causal_softmax_case(params):
    cases = {
        ((3, 3), None, None),
        ((32, 512), None, None),
    }

    return (
        _is_float32(params)
        and _shape_case(
            params,
            "shape",
            "input_strides",
            "out_strides",
        )
        in cases
    )


def _is_smoke_torch_op_case(params):
    op_meta = params.get("op_meta")

    return (
        isinstance(op_meta, dict)
        and op_meta.get("name") in _SMOKE_TORCH_OPS
        and params.get("shape") == (13, 4)
        and _is_float32(params)
    )


def _active_device_selectors_for_torch_device(config, torch_device):
    """Return platform or torch device names selected for a torch device type."""

    if not torch_device:
        return ()

    cli_devices = config.getoption("--devices") or ()
    requested_platforms = tuple(
        name
        for name in cli_devices
        if _PLATFORM_TO_TORCH_DEVICE.get(name) == torch_device
    )

    if requested_platforms:
        return requested_platforms

    # The pybind layer maps torch device names (e.g. "cuda") to the backend
    # compiled into the current wheel, avoiding probes of inactive CUDA siblings.

    return (torch_device,)


def _resolve_device(name):
    """Map a platform name (e.g., `ascend`) to a PyTorch device type (e.g., `npu`)."""

    return _PLATFORM_TO_TORCH_DEVICE.get(name, name)


def pytest_generate_tests(metafunc):
    already_parametrized = _get_parametrized_args(metafunc)

    if "dtype" in metafunc.fixturenames and "dtype" not in already_parametrized:
        metafunc.parametrize(
            "dtype, rtol, atol",
            (
                (torch.float32, 1e-7, 1e-7),
                (torch.float16, 1e-3, 1e-3),
                (torch.bfloat16, 1e-3, 1e-3),
            ),
        )

    if "device" in metafunc.fixturenames and "device" not in already_parametrized:
        cli_devices = metafunc.config.getoption("--devices")
        available = get_available_devices()

        if cli_devices:
            devices = tuple(
                d for d in (_resolve_device(x) for x in cli_devices) if d in available
            )
        else:
            devices = ()

        devices = devices or available

        # Joint `(device, implementation_index)` parametrize: generate only
        # pairs where the op has an active implementation on that device.
        # Avoids cross-device pollution — an impl active on `cpu` but not on
        # `npu` no longer appears as a runtime skip in the npu column.
        if (
            "implementation_index" in metafunc.fixturenames
            and "implementation_index" not in already_parametrized
        ):
            op_cls = _op_class_from_module(metafunc.module)

            if op_cls is not None and hasattr(op_cls, "active_implementation_indices"):
                pairs = [
                    (dev, idx)
                    for dev in devices
                    for idx in op_cls.active_implementation_indices(dev)
                ]

                if not pairs:
                    # Emit one skipped placeholder so test IDs read
                    # `[skip-dtype0-...]` instead of `[NOTSET-...]`.
                    # `get_available_devices()` always includes `"cpu"`, so
                    # `devices[0]` is safe.
                    pairs = [
                        pytest.param(
                            devices[0],
                            0,
                            marks=pytest.mark.skip(
                                reason=(
                                    f"{op_cls.__name__} has no active "
                                    "implementation on any available device"
                                )
                            ),
                            id="skip",
                        )
                    ]

                metafunc.parametrize("device, implementation_index", pairs)

                return

        metafunc.parametrize("device", devices)


def _op_class_from_module(module):
    """Derive the `infini.ops.<Op>` class from a `tests/test_<snake>.py` module.

    Test modules have already imported `infini.ops` by the time this runs, so
    no `try/except ImportError` is needed — a real import failure would have
    aborted collection long before.
    """
    module_name = module.__name__.rsplit(".", 1)[-1]

    if not module_name.startswith("test_"):
        return None

    op_snake = module_name[len("test_") :]
    op_pascal = "".join(part.capitalize() for part in op_snake.split("_"))

    import infini.ops as _ops

    return getattr(_ops, op_pascal, None)


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    if pyfuncitem.get_closest_marker("auto_act_and_assert"):
        func_kwargs = {
            arg: pyfuncitem.funcargs[arg] for arg in pyfuncitem._fixtureinfo.argnames
        }

        payload = pyfuncitem.obj(**func_kwargs)

        func = payload.func
        ref = payload.ref
        args = payload.args
        kwargs = payload.kwargs

        ref_args = _clone(args)
        ref_kwargs = _clone(kwargs)

        output = func(*args, **kwargs)
        expected = ref(*ref_args, **ref_kwargs)

        if pyfuncitem.config.getoption("--benchmark"):
            stmt = "func(*args, **kwargs)"
            profile_output = pyfuncitem.config._infini_host_range_profile_output
            timer_kwargs = (
                {"timer": time.perf_counter} if profile_output is not None else {}
            )

            func_timer = benchmark.Timer(
                stmt=stmt,
                globals={"func": func, "args": args, "kwargs": kwargs},
                label=func.__name__,
                description="InfiniOps",
                **timer_kwargs,
            )

            if profile_output is None:
                func_measurement = func_timer.blocked_autorange()
            else:
                import infini.ops as ops

                op_cls = _op_class_from_module(pyfuncitem.module)
                clear_cache = getattr(op_cls, "clear_cache", None)

                if callable(clear_cache):
                    clear_cache()

                params = _callspec_params(pyfuncitem)
                context = {
                    "nodeid": pyfuncitem.nodeid,
                    "operator": host_range_profile.operator_from_module(
                        pyfuncitem.module
                    ),
                    "backend": host_range_profile.backend_from_devices(
                        pyfuncitem.config.getoption("--devices"),
                        params.get("device"),
                    ),
                }

                def synchronize():
                    host_range_profile.synchronize_torch_device(
                        torch, params.get("device")
                    )

                _, cold_summaries = host_range_profile.collect_ranges(
                    ops._host_range_profile_start,
                    ops._host_range_profile_stop,
                    lambda: func(*args, **kwargs),
                    synchronize=synchronize,
                )
                host_range_profile.append_rows(
                    profile_output,
                    host_range_profile.expand_summary_rows(
                        cold_summaries, phase="cold", **context
                    ),
                )

                func_measurement = func_timer.blocked_autorange()
                warm_summaries = host_range_profile.collect_replayed_ranges(
                    ops._host_range_profile_start,
                    ops._host_range_profile_stop,
                    lambda: func(*args, **kwargs),
                    func_measurement,
                    synchronize=synchronize,
                )
                warm_rows = host_range_profile.expand_summary_rows(
                    warm_summaries, phase="warm", **context
                )
                warm_rows.append(
                    host_range_profile.measurement_row(
                        func_measurement, phase="warm", **context
                    )
                )
                host_range_profile.append_rows(profile_output, warm_rows)

            ref_timer = benchmark.Timer(
                stmt=stmt,
                globals={"func": ref, "args": ref_args, "kwargs": ref_kwargs},
                label=func.__name__,
                description="Reference",
                **timer_kwargs,
            )
            ref_measurement = ref_timer.blocked_autorange()

            benchmark.Compare((func_measurement, ref_measurement)).print()

        rtol = payload.rtol
        atol = payload.atol

        # `torch.allclose` rejects `bool` dtypes — use `torch.equal` for
        # non-floating outputs (bool, int) so comparison ops work. Pass
        # `equal_nan=True` so NaN-in-both-positions (common for special
        # functions fed out-of-domain inputs) does not fail the test.
        if output.dtype.is_floating_point:
            assert torch.allclose(
                output, expected, rtol=rtol, atol=atol, equal_nan=True
            )
        else:
            assert torch.equal(output, expected)

        return True


def _get_parametrized_args(metafunc):
    parametrized_args = set()

    for marker in metafunc.definition.iter_markers(name="parametrize"):
        args = marker.args[0]

        if isinstance(args, str):
            parametrized_args.update(x.strip() for x in args.split(","))
        elif isinstance(args, (list, tuple)):
            parametrized_args.update(args)

    return parametrized_args


def _test_case_path_from_request(request):
    return f"{_module_path_from_request(request)}::{request.node.name}"


def _module_path_from_request(request):
    return f"{request.module.__name__.replace('.', '/')}.py"


def _hash(string):
    return int(hashlib.sha256(string.encode("utf-8")).hexdigest(), 16) % 2**32


def _clone(obj):
    if isinstance(obj, torch.Tensor):
        return clone_strided(obj)

    if isinstance(obj, tuple):
        return tuple(_clone(a) for a in obj)

    if isinstance(obj, list):
        return [_clone(a) for a in obj]

    if isinstance(obj, dict):
        return {key: _clone(value) for key, value in obj.items()}

    return obj
