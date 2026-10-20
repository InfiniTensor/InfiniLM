#!/bin/bash
set -euo pipefail
ulimit -c 0
TASK_ROOT=/path/to/cluster-home/hygon-inference-refactor-20261014
OLD_ROOT=/path/to/cluster-home/hygon-inference-20261013
APP=/public/software/apps/apptainer/1.3.4/bin/apptainer
export HSA_FORCE_FINE_GRAIN_PCIE=1
node_local="/dev/shm/infini-pyc-${SLURM_JOB_ID}.${SLURM_STEP_ID}-${SLURM_PROCID}"
test ! -e "$node_local"
mkdir -m700 "$node_local"
local_fs=$(stat -f -c %T "$node_local")
test "$local_fs" = tmpfs
export LOCAL_PYCACHE_HOST_R3097="$node_local"
echo "LOCAL_PYCACHE_R3097 node=$(hostname) filesystem=$local_fs root=$node_local"
work_name="runtime-raw-sdk-graph-r2323-${SLURM_JOB_ID}.${SLURM_STEP_ID}-rank${SLURM_PROCID}"
task_scratch="$TASK_ROOT/results/$work_name"
mkdir "$task_scratch"
export APPTAINER_TMPDIR="$task_scratch"
export TMPDIR="$task_scratch"
export APPTAINERENV_TMPDIR=/tmp
if [ "$1" = glm ]; then
  export GLM_PIPELINE_TAIL_NODE="$(scontrol show hostnames "$SLURM_STEP_NODELIST" | tail -n1)"
  cp "$TASK_ROOT/incoming/resource_glm_r1780_inside.sh" "$task_scratch/job-inside.sh"
else
  cp "$TASK_ROOT/incoming/${WAVE_INSIDE_R2323}" "$task_scratch/job-inside.sh"
fi
echo "SHARED_CACHED_R1780_START node=$(hostname) step=${SLURM_JOB_ID}.${SLURM_STEP_ID} scratch=$work_name"
"$APP" exec --rocm --home "$OLD_ROOT/build-deps/root:/root" \
 --bind "$OLD_ROOT:/old:ro" --bind "$TASK_ROOT:/refactor" \
 --bind /opt/hyhal:/opt/hyhal:ro --bind /path/to/cluster-home/shared:/shared:ro \
 --bind "$task_scratch:/tmp" --bind "$node_local:/local" \
 "$TASK_ROOT/containers/dtk26-sandbox-r1" /bin/bash --noprofile --norc /tmp/job-inside.sh
echo "SHARED_CACHED_R1780_COMPLETE node=$(hostname) scratch=$work_name"
