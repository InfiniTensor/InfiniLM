#!/usr/bin/env bash
# Create isolated $REPO/.venv-vllm (does not modify system Python / HF stack).
# Optional --moe: install transformers>=5 in that venv for TransformersMoEForCausalLM.
# Optional --tsinghua: use Tsinghua tuna PyPI mirror (faster in CN).
set -euo pipefail

MOE=0
TSINGHUA=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --moe)
      MOE=1
      shift
      ;;
    --tsinghua)
      TSINGHUA=1
      shift
      ;;
    -h | --help)
      echo "Usage: $0 [--moe] [--tsinghua] [REPO]"
      echo "  REPO  Workspace root (default: parent of InfiniLM/)."
      echo "  --moe Also pip install 'transformers>=5.0.0,<6' for vLLM MoE fallback."
      echo "  --tsinghua Use https://pypi.tuna.tsinghua.edu.cn/simple (Tsinghua mirror)."
      echo "HF parity (transformers 4.57.1) uses a separate venv: bash InfiniLM/examples/setup_hf_parity_venv.sh"
      echo "Env: VLLM_PIN=pinned version (default: 0.19.0)."
      echo "     VLLM_PIP_INDEX_URL / VLLM_PIP_TRUSTED_HOST  custom index (if set, used instead of --tsinghua)."
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

VLLM_PIN="${VLLM_PIN:-0.19.0}"
VENV="${REPO}/.venv-vllm"

if [[ ! -d "${REPO}" ]]; then
  echo "REPO is not a directory: ${REPO}" >&2
  exit 1
fi

echo "REPO=${REPO}"
echo "VENV=${VENV}"

PIP_FLAGS=()
if [[ -n "${VLLM_PIP_INDEX_URL:-}" ]]; then
  echo "pip index: ${VLLM_PIP_INDEX_URL}"
  _host="${VLLM_PIP_TRUSTED_HOST:-}"
  if [[ -z "${_host}" ]]; then
    _u="${VLLM_PIP_INDEX_URL#*://}"
    _host="${_u%%/*}"
  fi
  PIP_FLAGS=(-i "${VLLM_PIP_INDEX_URL}" --trusted-host "${_host}")
elif [[ "${TSINGHUA}" -eq 1 ]]; then
  echo "pip index: Tsinghua tuna (pypi.tuna.tsinghua.edu.cn)"
  PIP_FLAGS=(
    -i "https://pypi.tuna.tsinghua.edu.cn/simple"
    --trusted-host "pypi.tuna.tsinghua.edu.cn"
  )
fi

if [[ ! -d "${VENV}" ]]; then
  python3 -m venv "${VENV}"
fi
# shellcheck source=/dev/null
source "${VENV}/bin/activate"
python -m pip install "${PIP_FLAGS[@]}" -U pip wheel setuptools
python -m pip install "${PIP_FLAGS[@]}" "vllm==${VLLM_PIN}"

if [[ "${MOE}" -eq 1 ]]; then
  echo "Installing transformers>=5 for TransformersMoEForCausalLM (MoE fallback)..."
  python -m pip install "${PIP_FLAGS[@]}" 'transformers>=5.0.0,<6'
fi

python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "Activate with:"
echo "  source \"${VENV}/bin/activate\""
