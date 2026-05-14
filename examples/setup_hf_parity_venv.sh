#!/usr/bin/env bash
# Create $REPO/.venv-no-vllm: HF parity (transformers==4.57.1) separate from .venv-vllm (vLLM + TF5).
# Uses --system-site-packages so the venv reuses the container / base install CUDA torch (no second torch wheel).
#
# Usage:
#   bash InfiniLM/examples/setup_hf_parity_venv.sh [--tsinghua] [REPO]
set -euo pipefail

TSINGHUA=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tsinghua)
      TSINGHUA=1
      shift
      ;;
    -h | --help)
      echo "Usage: $0 [--tsinghua] [REPO]"
      echo "  Creates \$REPO/.venv-no-vllm with transformers==4.57.1 (HF bench / parity)."
      exit 0
      ;;
    *)
      break
      ;;
  esac
done

REPO="${1:-}"
if [[ -z "${REPO}" ]]; then
  REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi

VENV="${REPO}/.venv-no-vllm"
REQ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/requirements-hf-parity.txt"

if [[ ! -d "${REPO}" ]]; then
  echo "REPO is not a directory: ${REPO}" >&2
  exit 1
fi
if [[ ! -f "${REQ}" ]]; then
  echo "Missing requirements file: ${REQ}" >&2
  exit 1
fi

echo "REPO=${REPO}"
echo "VENV=${VENV}"

PIP_FLAGS=()
if [[ -n "${HF_PARITY_PIP_INDEX_URL:-}" ]]; then
  _host="${HF_PARITY_PIP_TRUSTED_HOST:-}"
  if [[ -z "${_host}" ]]; then
    _u="${HF_PARITY_PIP_INDEX_URL#*://}"
    _host="${_u%%/*}"
  fi
  PIP_FLAGS=(-i "${HF_PARITY_PIP_INDEX_URL}" --trusted-host "${_host}")
elif [[ "${TSINGHUA}" -eq 1 ]]; then
  PIP_FLAGS=(
    -i "https://pypi.tuna.tsinghua.edu.cn/simple"
    --trusted-host "pypi.tuna.tsinghua.edu.cn"
  )
fi

if [[ ! -d "${VENV}" ]]; then
  python3 -m venv --system-site-packages "${VENV}"
fi
# shellcheck source=/dev/null
source "${VENV}/bin/activate"
python -m pip install "${PIP_FLAGS[@]}" -U pip wheel setuptools
python -m pip install "${PIP_FLAGS[@]}" -r "${REQ}"

python -c "import transformers, torch; print('transformers', transformers.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "OK: activate with: source \"${VENV}/bin/activate\""
