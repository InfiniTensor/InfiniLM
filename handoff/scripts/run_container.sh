#!/bin/bash
set -euo pipefail
umask 077
: "${TASK_ROOT:?}" "${OLD_ROOT:?}" "${MODEL_ROOT:?}" "${CONTAINER_IMAGE:?}" "${BUILD_HOME:?}" "${HANDOFF_REPO:?}"
test "$#" -gt 0
test -d "$TASK_ROOT" && test -d "$OLD_ROOT" && test -d "$MODEL_ROOT"
test -d "$BUILD_HOME" && test -d "$HANDOFF_REPO/handoff"
local_root=$(mktemp -d /dev/shm/infini-handoff.XXXXXXXX)
test "$(stat -f -c %T "$local_root")" = tmpfs
mkdir "$local_root/tmp" "$local_root/cache"
cleanup() {
  # The directory is unique to this invocation. Leave caches for diagnosis.
  printf 'Local scratch retained: %s\n' "$local_root"
}
trap cleanup EXIT
export APPTAINERENV_PYTHONPYCACHEPREFIX=/local/cache
export APPTAINERENV_TMPDIR=/local/tmp
export APPTAINERENV_OMP_NUM_THREADS=1
args=(exec --rocm --home "$BUILD_HOME:/root"
  --bind "$TASK_ROOT:/refactor" --bind "$OLD_ROOT:/old:ro"
  --bind "$MODEL_ROOT:/shared:ro" --bind "$local_root:/local"
  --bind "$HANDOFF_REPO:/handoff-repo:ro")
if [ -n "${HYHAL_ROOT:-}" ]; then args+=(--bind "$HYHAL_ROOT:/opt/hyhal:ro"); fi
"${APPTAINER:-apptainer}" "${args[@]}" "$CONTAINER_IMAGE" "$@"
