import functools
import hashlib
import importlib.util
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import triton
import triton.backends

_OPS_DIR = Path(__file__).resolve().parent.parent / "ops"

_DRIVER_BACKENDS = {
    "cuda": "nvidia",
}


def _cache_directory():
    configured_path = os.environ.get("INFINI_OPS_TRITON_CACHE_DIR")
    if configured_path:
        return Path(configured_path)

    if os.name == "nt":
        base_path = os.environ.get("LOCALAPPDATA")
        if base_path:
            return Path(base_path) / "infiniops" / "triton"
    else:
        base_path = os.environ.get("XDG_CACHE_HOME")
        if base_path:
            return Path(base_path) / "infiniops" / "triton"

        home_path = os.environ.get("HOME")
        if home_path:
            return Path(home_path) / ".cache" / "infiniops" / "triton"

    return Path(tempfile.gettempdir()) / "infiniops" / "triton"


def _cache_path(identity, suffix=""):
    hash_value = 14695981039346656037
    for byte in identity.encode():
        hash_value ^= byte
        hash_value = hash_value * 1099511628211 & 0xFFFFFFFFFFFFFFFF

    return _cache_directory() / f"{hash_value:016x}{suffix}"


def _write_json_atomically(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as temporary_file:
            json.dump(value, temporary_file, sort_keys=True)
            temporary_path = Path(temporary_file.name)

        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def get_kernel_artifact_prefix(identity):
    return str(_cache_path(identity))


def read_kernel_artifact(output_prefix, expected_identity):
    try:
        metadata = json.loads(Path(f"{output_prefix}.json").read_text(encoding="utf-8"))
        if metadata["cache_identity"] != expected_identity:
            return None

        name = metadata["name"]
        if not isinstance(name, str) or not name:
            return None

        num_warps = metadata["num_warps"]
        if not _is_int32(num_warps, unsigned=True) or num_warps == 0:
            return None

        binary_extension = metadata["binary_ext"]
        if not isinstance(binary_extension, str) or not binary_extension.isalnum():
            return None

        shared_memory_size = metadata["shared"]
        if not _is_int32(shared_memory_size, unsigned=True):
            return None

        global_scratch_size = metadata["global_scratch_size"]
        profile_scratch_size = metadata["profile_scratch_size"]
        if not _is_int32(global_scratch_size) or not _is_int32(profile_scratch_size):
            return None

        binary = Path(f"{output_prefix}.{binary_extension}").read_bytes()
        if not binary:
            return None

        return {
            "name": name,
            "num_warps": num_warps,
            "binary": binary,
            "shared_memory_size": shared_memory_size,
            "global_scratch_size": global_scratch_size,
            "profile_scratch_size": profile_scratch_size,
        }
    except (KeyError, OSError, TypeError, ValueError):
        return None


def read_auto_tuning_config(identity):
    try:
        document = json.loads(
            _cache_path(identity, ".autotune").read_text(encoding="utf-8")
        )
        if document["cache_identity"] != identity:
            return None

        config = document["config"]
        if (
            not _is_int32(config["num_warps"], unsigned=True)
            or config["num_warps"] == 0
        ):
            return None
        if (
            not _is_int32(config["num_stages"], unsigned=True)
            or config["num_stages"] == 0
        ):
            return None
        if not isinstance(config["constexprs"], dict):
            return None
        if not all(
            isinstance(name, str) and _is_int32(value)
            for name, value in config["constexprs"].items()
        ):
            return None

        return config
    except (KeyError, OSError, TypeError, ValueError):
        return None


def _is_int32(value, *, unsigned=False):
    minimum = 0 if unsigned else -(1 << 31)
    return type(value) is int and minimum <= value < (1 << 31)


def write_auto_tuning_config(identity, config):
    _write_json_atomically(
        _cache_path(identity, ".autotune"),
        {"cache_identity": identity, "config": config},
    )


def _get_kernel_source_path(op_name):
    source_path = _OPS_DIR / op_name / "jit.py"

    if not source_path.is_file():
        raise FileNotFoundError(f"The Triton kernel `{source_path}` does not exist.")

    return source_path


def get_compilation_fingerprint(op_name):
    kernel = _load_kernel(op_name)
    digest = hashlib.sha256()
    digest.update(f"triton={triton.__version__}\n".encode())
    digest.update(b"compiler\0")
    digest.update(Path(__file__).read_bytes())
    digest.update(b"triton-jit\0")
    digest.update(kernel.cache_key.encode())

    return digest.hexdigest()


@functools.cache
def _load_kernel(op_name):
    source_path = _get_kernel_source_path(op_name)
    module_name = f"infini.triton.ops.{op_name}.jit"
    module_spec = importlib.util.spec_from_file_location(module_name, source_path)

    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"The Triton kernel `{source_path}` could not be loaded.")

    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    kernel = getattr(module, "kernel")

    while not isinstance(kernel, triton.runtime.JITFunction):
        kernel = kernel.fn

    return kernel


def _create_target(target):
    backend = target["backend"]
    architecture = target["architecture"]
    if backend == "cuda":
        architecture = int(architecture)

    return triton.backends.compiler.GPUTarget(
        backend, architecture, target["warp_size"]
    )


@contextmanager
def _use_device(target, device_id, stream):
    previous_driver = triton.runtime.driver.active
    previous_device = previous_driver.get_current_device()
    driver_backend = _DRIVER_BACKENDS[target["backend"]]
    active_driver = triton.backends.backends[driver_backend].driver()

    try:
        triton.runtime.driver.set_active(active_driver)
        active_driver.set_current_device(device_id)
        device_interface = active_driver.get_device_interface()

        if stream:
            benchmark_stream = device_interface.ExternalStream(stream, device=device_id)
        else:
            benchmark_stream = device_interface.default_stream(device_id)

        with device_interface.stream(benchmark_stream):
            yield
    finally:
        triton.runtime.driver.set_active(previous_driver)
        previous_driver.set_current_device(previous_device)


def _build_ast_source(kernel, runtime_signature, constexprs):
    runtime_types = (
        [part.strip() for part in runtime_signature.split(",")]
        if runtime_signature
        else []
    )
    constexpr_names = {
        name
        for name, parameter in zip(kernel.arg_names, kernel.params)
        if parameter.is_constexpr
    }
    missing_constexprs = constexpr_names - constexprs.keys()
    if missing_constexprs:
        names = ", ".join(sorted(missing_constexprs))
        raise ValueError(f"The config does not define constexprs: {names}.")

    unknown_constexprs = constexprs.keys() - constexpr_names
    if unknown_constexprs:
        names = ", ".join(sorted(unknown_constexprs))
        raise ValueError(f"The config defines unknown constexprs: {names}.")

    expected_runtime_count = len(kernel.arg_names) - len(constexpr_names)
    if len(runtime_types) != expected_runtime_count:
        raise ValueError(
            f"The runtime signature has {len(runtime_types)} entries, but `kernel` "
            f"has {expected_runtime_count} runtime parameters."
        )

    signature_types = {}
    constants = {}
    attributes = {}

    runtime_type_iterator = iter(runtime_types)
    for index, (name, parameter) in enumerate(zip(kernel.arg_names, kernel.params)):
        if parameter.is_constexpr:
            constants[(index,)] = constexprs[name]
            signature_types[name] = "constexpr"
            continue

        runtime_type = next(runtime_type_iterator)
        if runtime_type.endswith(":1"):
            constants[(index,)] = 1
            signature_types[name] = "constexpr"
        elif runtime_type.endswith(":16"):
            signature_types[name] = runtime_type[:-3]
            attributes[(index,)] = [["tt.divisibility", 16]]
        else:
            signature_types[name] = runtime_type

    return triton.compiler.ASTSource(
        fn=kernel,
        signature=signature_types,
        constexprs=constants,
        attrs=attributes,
    )


def _compile_loaded_kernel(kernel, signature, target, config):
    source = _build_ast_source(kernel, signature, config["constexprs"])

    return triton.compile(
        source,
        target=target,
        options={
            "num_warps": config["num_warps"],
            "num_stages": config["num_stages"],
        },
    )


def _write_artifacts(compiled_kernel, target, output_prefix, cache_identity):
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    binary_extension = triton.compiler.make_backend(target).binary_ext
    binary_path = Path(f"{output_prefix}.{binary_extension}")
    metadata_path = Path(f"{output_prefix}.json")
    metadata = {
        "name": compiled_kernel.metadata.name,
        "num_warps": compiled_kernel.metadata.num_warps,
        "binary_ext": binary_extension,
        "shared": compiled_kernel.metadata.shared,
        "global_scratch_size": compiled_kernel.metadata.global_scratch_size,
        "profile_scratch_size": compiled_kernel.metadata.profile_scratch_size,
        "cache_identity": cache_identity,
    }
    temporary_paths = []

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=output_prefix.parent, delete=False
        ) as binary_file:
            binary_file.write(compiled_kernel.asm[binary_extension])
            binary_temporary_path = Path(binary_file.name)
            temporary_paths.append(binary_temporary_path)

        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output_prefix.parent, delete=False
        ) as metadata_file:
            json.dump(metadata, metadata_file, sort_keys=True)
            metadata_temporary_path = Path(metadata_file.name)
            temporary_paths.append(metadata_temporary_path)

        os.replace(binary_temporary_path, binary_path)
        os.replace(metadata_temporary_path, metadata_path)
    finally:
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)


def compile_kernel(op_name, output_prefix, signature, config, target, cache_identity):
    kernel = _load_kernel(op_name)
    compilation_target = _create_target(target)
    compiled_kernel = _compile_loaded_kernel(
        kernel, signature, compilation_target, config
    )
    _write_artifacts(compiled_kernel, compilation_target, output_prefix, cache_identity)


def _build_launch_arguments(kernel, runtime_arguments, constexprs):
    runtime_argument_iterator = iter(runtime_arguments)
    launch_arguments = []
    sentinel = object()

    for name, parameter in zip(kernel.arg_names, kernel.params):
        if parameter.is_constexpr:
            if name not in constexprs:
                raise ValueError(f"The config does not define constexpr `{name}`.")
            launch_arguments.append(constexprs[name])
            continue

        try:
            launch_arguments.append(next(runtime_argument_iterator))
        except StopIteration:
            raise ValueError(
                f"No runtime argument was provided for `{name}`."
            ) from None

    if next(runtime_argument_iterator, sentinel) is not sentinel:
        raise ValueError("More runtime arguments were provided than expected.")

    return launch_arguments


def _benchmark_candidate(
    kernel,
    compiled_kernel,
    arguments,
    candidate,
    warmup_milliseconds,
    repetition_milliseconds,
):
    kernel_call = functools.partial(
        compiled_kernel[tuple(candidate["grid"])],
        *_build_launch_arguments(kernel, arguments, candidate["constexprs"]),
    )

    return triton.testing.do_bench(
        kernel_call,
        warmup=warmup_milliseconds,
        rep=repetition_milliseconds,
        return_mode="median",
    )


def auto_tune(
    op_name,
    candidates,
    arguments,
    stream,
    warmup_milliseconds,
    repetition_milliseconds,
    target,
    device_id,
):
    if not candidates:
        raise ValueError("At least one auto-tuning candidate is required.")

    if warmup_milliseconds < 0:
        raise ValueError("The auto-tuning warmup duration must not be negative.")

    if repetition_milliseconds <= 0:
        raise ValueError("The auto-tuning repetition duration must be positive.")

    kernel = _load_kernel(op_name)
    compilation_target = _create_target(target)
    best_index = None
    best_time = float("inf")

    with _use_device(target, device_id, stream):
        for index, candidate in enumerate(candidates):
            try:
                compiled_kernel = _compile_loaded_kernel(
                    kernel,
                    candidate["signature"],
                    compilation_target,
                    candidate,
                )
                elapsed_time = _benchmark_candidate(
                    kernel,
                    compiled_kernel,
                    arguments,
                    candidate,
                    warmup_milliseconds,
                    repetition_milliseconds,
                )
                _write_artifacts(
                    compiled_kernel,
                    compilation_target,
                    candidate["output_prefix"],
                    candidate["cache_identity"],
                )
            except (triton.CompilationError, triton.OutOfResources):
                continue

            if elapsed_time < best_time:
                best_time = elapsed_time
                best_index = index

    if best_index is None:
        raise RuntimeError("No auto-tuning candidate completed successfully.")

    return best_index
