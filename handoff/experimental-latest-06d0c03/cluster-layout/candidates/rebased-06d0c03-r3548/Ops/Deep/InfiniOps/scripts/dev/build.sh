#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${repo_root}/scripts/dev/platforms.sh"

usage() {
    cat <<'EOF'
Usage:
  scripts/dev/build.sh [cpu|nvidia|iluvatar|hygon|thead|metax|moore|cambricon|ascend|auto] [--jobs N] [--smoke]

Examples:
  scripts/dev/build.sh
  scripts/dev/build.sh cambricon
  scripts/dev/build.sh metax --jobs 8
  scripts/dev/build.sh nvidia
  scripts/dev/build.sh nvidia --smoke
  scripts/dev/build.sh auto

What it does:
  1. Re-runs CMake configure in a persistent build directory.
  2. Incrementally builds the Python extension target (`ops`).
  3. Installs the result into `build-<platform>/install/infini/`.

Why this is faster than `pip install .[dev] --no-build-isolation`:
  - It reuses the same build directory, so unchanged objects are not rebuilt.
  - It avoids wheel build/install work in a temp directory on every run.
  - `[dev]` dependencies only need to be installed once.

Important:
  - This repo's wrapper/codegen runs during CMake configure, not only build.
    So this script always runs `cmake -S -B ...` first, ensuring generator
    changes (`generate_torch_ops.py`, `generate_wrappers.py`, YAML edits) are
    picked up before the incremental build.
EOF
}

platform="auto"
jobs="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"
torch_jobs="${INFINI_OPS_TORCH_COMPILE_JOBS:-2}"
binding_jobs="${INFINI_OPS_BINDING_COMPILE_JOBS:-2}"
smoke_build="OFF"

while (($#)); do
    if infini_ops_is_supported_platform "$1"; then
        platform="$1"
        shift
        continue
    fi

    case "$1" in
        auto)
            platform="$1"
            shift
            ;;
        -j|--jobs)
            jobs="$2"
            shift 2
            ;;
        --torch-jobs)
            torch_jobs="$2"
            shift 2
            ;;
        --binding-jobs)
            binding_jobs="$2"
            shift 2
            ;;
        --smoke)
            smoke_build="ON"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

python_bin="${PYTHON_BIN:-$(command -v python3)}"

if [[ "$platform" == "auto" ]]; then
    mapfile -t detected_platforms < <(infini_ops_detect_platforms)

    case "${#detected_platforms[@]}" in
        0)
            platform="cpu"
            echo "[build] auto-detect: no accelerator platform found, using cpu"
            ;;
        1)
            platform="${detected_platforms[0]}"
            echo "[build] auto-detect: using ${platform}"
            ;;
        *)
            echo "Auto-detected multiple accelerator platforms: ${detected_platforms[*]}." >&2
            echo "Pass one explicitly: $(infini_ops_supported_platforms_csv)." >&2
            exit 1
            ;;
    esac
fi

with_cpu="ON"
with_torch="ON"
with_nvidia="OFF"
with_iluvatar="OFF"
with_hygon="OFF"
with_thead="OFF"
with_cambricon="OFF"
with_metax="OFF"
with_moore="OFF"
with_ascend="OFF"

case "$platform" in
    nvidia)
        with_nvidia="ON"
        ;;
    iluvatar)
        with_iluvatar="ON"
        ;;
    hygon)
        with_hygon="ON"
        ;;
    thead)
        with_thead="ON"
        ;;
    cambricon)
        with_cambricon="ON"
        ;;
    metax)
        with_metax="ON"
        ;;
    moore)
        with_moore="ON"
        ;;
    ascend)
        with_ascend="ON"
        ;;
    cpu)
        ;;
    *)
        echo "Unsupported platform: $platform" >&2
        exit 1
        ;;
esac

build_dir="${repo_root}/build-${platform}"
install_root="${build_dir}/install"
install_dir="${install_root}/infini"
generator="${CMAKE_GENERATOR:-}"
cached_generator=""

if [[ -z "${generator}" && -f "${build_dir}/CMakeCache.txt" ]]; then
    cached_generator="$(sed -n 's/^CMAKE_GENERATOR:INTERNAL=//p' "${build_dir}/CMakeCache.txt" | head -n 1)"
    generator="${cached_generator}"
fi

if [[ "${generator}" == "Ninja" ]] && ! command -v ninja > /dev/null 2>&1; then
    generator=""
fi

if [[ -z "${generator}" ]]; then
    if command -v ninja > /dev/null 2>&1; then
        generator="Ninja"
    else
        generator="Unix Makefiles"
    fi
fi

if [[ -n "${cached_generator}" && "${cached_generator}" != "${generator}" ]]; then
    echo "[build] generator changed: ${cached_generator} -> ${generator}"
    rm -f "${build_dir}/CMakeCache.txt"
    rm -rf "${build_dir}/CMakeFiles"
fi

echo "[build] repo      : ${repo_root}"
echo "[build] platform  : ${platform}"
echo "[build] python    : ${python_bin}"
echo "[build] build dir : ${build_dir}"
echo "[build] install   : ${install_root}"
echo "[build] package   : ${install_dir}"
echo "[build] generator : ${generator}"
echo "[build] jobs      : build=${jobs} torch=${torch_jobs} binding=${binding_jobs}"
echo "[build] smoke     : ${smoke_build}"

cmake -S "${repo_root}" -B "${build_dir}" -G "${generator}" \
    -DPython_EXECUTABLE="${python_bin}" \
    -DWITH_CPU="${with_cpu}" \
    -DWITH_TORCH="${with_torch}" \
    -DWITH_NVIDIA="${with_nvidia}" \
    -DWITH_ILUVATAR="${with_iluvatar}" \
    -DWITH_HYGON="${with_hygon}" \
    -DWITH_THEAD="${with_thead}" \
    -DWITH_CAMBRICON="${with_cambricon}" \
    -DWITH_METAX="${with_metax}" \
    -DWITH_MOORE="${with_moore}" \
    -DWITH_ASCEND="${with_ascend}" \
    -DAUTO_DETECT_DEVICES=OFF \
    -DAUTO_DETECT_BACKENDS=OFF \
    -DGENERATE_PYTHON_BINDINGS=ON \
    -DINFINI_OPS_SMOKE_BUILD="${smoke_build}" \
    -DINFINI_OPS_TORCH_COMPILE_JOBS="${torch_jobs}" \
    -DINFINI_OPS_BINDING_COMPILE_JOBS="${binding_jobs}"

cmake --build "${build_dir}" --target ops -j "${jobs}"
mkdir -p "${install_root}" "${install_dir}"
rm -f \
    "${install_root}/__init__.py" \
    "${install_root}/libinfiniops.so" \
    "${install_root}/torch_ops_metadata.json" \
    "${install_root}"/ops*.so
mkdir -p "${install_dir}"
cmake --install "${build_dir}" --prefix "${install_dir}"

cat <<EOF

[build] done

Use this build in pytest with:
  PYTHONPATH="${install_root}:\$PYTHONPATH" pytest ...

For example:
  PYTHONPATH="${install_root}:\$PYTHONPATH" pytest --devices ${platform}

Installed files:
  ${install_dir}/ops*.so
  ${install_dir}/torch_ops_metadata.json
EOF
