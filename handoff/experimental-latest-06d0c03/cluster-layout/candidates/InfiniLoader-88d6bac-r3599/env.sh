#!/usr/bin/env bash

_ILOADER_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case ":${PYTHONPATH:-}:" in
  *":${_ILOADER_ENV_DIR}/src:"*) ;;
  *) export PYTHONPATH="${_ILOADER_ENV_DIR}/src:${PYTHONPATH:-}" ;;
esac

iloader() {
  "${ILOADER_PYTHON:-python3}" -m infiniloader "$@"
}

if [ -n "${BASH_VERSION:-}" ] && command -v complete >/dev/null 2>&1; then
  eval "$("${ILOADER_PYTHON:-python3}" -m infiniloader completion bash 2>/dev/null || true)"
fi

unset _ILOADER_ENV_DIR
