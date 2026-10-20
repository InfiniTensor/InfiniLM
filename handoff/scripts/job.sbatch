#!/bin/bash
#SBATCH --job-name=infini-handoff
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=dcu:8
#SBATCH --time=04:00:00
#SBATCH --output=handoff-%j-%t.log
set -euo pipefail
: "${HANDOFF_REPO:?}" "${TASK_ROOT:?}"
mkdir -p "$TASK_ROOT/results" "$TASK_ROOT/logs" "$TASK_ROOT/control"
export GLM_PIPELINE_TAIL_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tail -n1)
# The caller selects a reviewed command explicitly. Default is hardware preflight.
if [ "$#" -eq 0 ]; then
  srun --kill-on-bad-exit=1 bash "$HANDOFF_REPO/handoff/scripts/run_container.sh" \
    bash -lc 'source /opt/dtk/env.sh; python /handoff-repo/handoff/scripts/preflight.py --output /refactor/results/preflight-${SLURM_JOB_ID}-${SLURM_PROCID}.json'
else
  srun --kill-on-bad-exit=1 bash "$HANDOFF_REPO/handoff/scripts/run_container.sh" "$@"
fi
