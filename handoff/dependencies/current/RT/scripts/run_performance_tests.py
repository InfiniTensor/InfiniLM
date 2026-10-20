import argparse
import json
import os
import pathlib
import platform
import subprocess
import sys

_BACKEND_OPTIONS = (
    ("WITH_NVIDIA", "nvidia"),
    ("WITH_ILUVATAR", "iluvatar"),
    ("WITH_HYGON", "hygon"),
    ("WITH_METAX", "metax"),
    ("WITH_MARS", "mars"),
    ("WITH_MOORE", "moore"),
    ("WITH_CAMBRICON", "cambricon"),
    ("WITH_ASCEND", "ascend"),
    ("WITH_CPU", "cpu"),
)


def _repo_root():
    return pathlib.Path(__file__).resolve().parents[1]


def _run_git(args):
    try:
        return subprocess.check_output(
            ["git", *args], cwd=_repo_root(), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _read_cmake_cache(build_dir):
    cache_path = build_dir / "CMakeCache.txt"
    values = {}
    if not cache_path.exists():
        return values

    for line in cache_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(("//", "#")) or "=" not in line:
            continue
        key_type, value = line.split("=", 1)
        key = key_type.split(":", 1)[0]
        values[key] = value
    return values


def _detect_backend(cache):
    for option, backend in _BACKEND_OPTIONS:
        if cache.get(option) == "ON":
            return backend
    return "unknown"


def _find_executables(build_dir, names):
    candidates = []
    perf_dir = build_dir / "tests" / "performance"
    search_roots = [
        perf_dir,
        perf_dir / "Release",
        perf_dir / "RelWithDebInfo",
        perf_dir / "Debug",
        build_dir,
    ]
    suffixes = [".exe"] if os.name == "nt" else [""]

    for name in names:
        found = None
        for root in search_roots:
            for suffix in suffixes:
                path = root / f"{name}{suffix}"
                if path.exists():
                    found = path
                    break
            if found is not None:
                break
        if found is None:
            raise FileNotFoundError(
                f"could not find performance executable {name!r} under {build_dir}"
            )
        candidates.append(found)

    return candidates


def _parse_json_lines(stdout, executable):
    results = []
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            results.append(json.loads(stripped))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{executable} printed an invalid JSON result line: {stripped}"
            ) from exc
    return results


def _run_executable(path):
    completed = subprocess.run(
        [str(path)],
        cwd=path.parent,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.stderr:
        sys.stderr.write(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"{path} exited with status {completed.returncode}")
    return _parse_json_lines(completed.stdout, path)


def _augment_result(result, metadata, backend_override, detected_backend):
    merged = dict(result)
    merged.update(metadata)
    if backend_override is not None:
        merged["backend"] = backend_override
    elif "backend" not in merged or merged["backend"] == "unknown":
        merged["backend"] = detected_backend
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", required=True, type=pathlib.Path)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument(
        "--test",
        action="append",
        dest="tests",
        default=None,
        help="Performance executable name to run. Can be passed more than once.",
    )
    args = parser.parse_args()

    build_dir = args.build_dir.resolve()
    cache = _read_cmake_cache(build_dir)
    detected_backend = _detect_backend(cache)
    tests = args.tests or [
        "perf_runtime_dispatch",
        "perf_memory",
        "perf_tensor_view",
        "perf_tensor_view_footprint",
    ]

    metadata = {
        "commit": _run_git(["rev-parse", "HEAD"]),
        "compiler": cache.get("CMAKE_CXX_COMPILER", "unknown"),
        "build_type": cache.get("CMAKE_BUILD_TYPE", "unknown"),
        "system": platform.platform(),
    }

    executables = _find_executables(build_dir, tests)
    results = []
    for executable in executables:
        for result in _run_executable(executable):
            results.append(
                _augment_result(result, metadata, args.backend, detected_backend)
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(results)} performance results to {args.output}")


if __name__ == "__main__":
    main()
