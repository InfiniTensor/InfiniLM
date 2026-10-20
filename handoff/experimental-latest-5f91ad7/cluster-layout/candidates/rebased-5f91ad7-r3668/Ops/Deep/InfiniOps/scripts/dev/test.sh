#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${repo_root}/scripts/dev/platforms.sh"

usage() {
    cat <<'EOF'
Usage:
  scripts/dev/test.sh [cpu|nvidia|iluvatar|hygon|thead|metax|moore|cambricon|ascend|auto] [--report PATH] [--no-build] [--smoke] [-- pytest-args...]

Examples:
  scripts/dev/test.sh cambricon
  scripts/dev/test.sh metax
  scripts/dev/test.sh nvidia
  scripts/dev/test.sh nvidia --smoke
  scripts/dev/test.sh cambricon -- tests/test_torch_ops.py -k index

Default behavior:
  1. Run `scripts/dev/build.sh <platform>`
  2. Run pytest with `--devices <platform> --report reports/<platform>.json`
  3. Print case-level and operator-level skip stats

Notes:
  - `--no-build` skips the build step and reuses the last local install.
  - `--smoke` reuses the upstream smoke subset (`INFINI_OPS_SMOKE_BUILD=ON`
    plus `pytest -m smoke`).
  - Extra args after `--` are passed straight to pytest.
EOF
}

platform="auto"
report_path=""
do_build=1
smoke_mode=0
pytest_args=()
pytest_targets=()
python_bin="${PYTHON_BIN:-$(command -v python3)}"

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
        --report)
            report_path="$2"
            shift 2
            ;;
        --no-build)
            do_build=0
            shift
            ;;
        --smoke)
            smoke_mode=1
            shift
            ;;
        --)
            shift
            pytest_args=("$@")
            break
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

if [[ "$platform" == "auto" ]]; then
    mapfile -t detected_platforms < <(infini_ops_detect_platforms)

    case "${#detected_platforms[@]}" in
        0)
            platform="cpu"
            echo "[test] auto-detect: no accelerator platform found, using cpu"
            ;;
        1)
            platform="${detected_platforms[0]}"
            echo "[test] auto-detect: using ${platform}"
            ;;
        *)
            echo "Auto-detected multiple accelerator platforms: ${detected_platforms[*]}." >&2
            echo "Pass one explicitly: $(infini_ops_supported_platforms_csv)." >&2
            exit 1
            ;;
    esac
fi

if [[ -z "$report_path" ]]; then
    report_path="${repo_root}/reports/${platform}.json"
fi

install_dir="${repo_root}/build-${platform}/install"
summary_script="${repo_root}/scripts/report/summarize.py"
detail_path="${report_path%.json}.details.jsonl"
text_path="${report_path%.json}.summary.txt"

if [[ "$do_build" -eq 1 ]]; then
    build_args=("$platform")

    if [[ "$smoke_mode" -eq 1 ]]; then
        build_args+=("--smoke")
    fi

    PYTHON_BIN="${python_bin}" "${repo_root}/scripts/dev/build.sh" "${build_args[@]}"
fi

if [[ ! -d "$install_dir" ]]; then
    echo "Install dir not found: $install_dir" >&2
    echo "Run scripts/dev/build.sh $platform first." >&2
    exit 1
fi

mkdir -p "$(dirname "$report_path")"
rm -f "$report_path" "$detail_path" "$text_path"

for arg in "${pytest_args[@]}"; do
    candidate="${arg%%::*}"

    if [[ -e "$candidate" || -e "${repo_root}/$candidate" ]]; then
        pytest_targets+=("$arg")
    fi
done

if [[ ${#pytest_targets[@]} -eq 0 ]]; then
    pytest_targets=("tests")
fi

pytest_extra_args=()
if [[ "$smoke_mode" -eq 1 ]]; then
    pytest_extra_args=(-m smoke)
fi

echo "[test] platform : ${platform}"
echo "[test] report   : ${report_path}"
echo "[test] install  : ${install_dir}"
echo "[test] targets  : ${pytest_targets[*]}"

set +e
PYTHONPATH="${install_dir}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${python_bin}" -m pytest \
    "${pytest_targets[@]}" \
    "${pytest_extra_args[@]}" \
    --devices "${platform}" \
    --report "${report_path}" \
    "${pytest_args[@]}"
pytest_status=$?
set -e

if [[ -f "$report_path" ]]; then
    echo ""
    "${python_bin}" "$summary_script" "$report_path"
else
    echo "[test] report not found: ${report_path}" >&2
fi

exit "$pytest_status"
