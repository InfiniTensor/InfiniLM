#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export PYTHONPYCACHEPREFIX="/local/health-${SLURM_JOB_ID}-${SLURM_STEP_ID}"
unset LD_PRELOAD
python - <<'PY'
import torch,json,pathlib,os,datetime
torch.set_num_threads(1)
r=pathlib.Path('/refactor');tag=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];q={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':False,'cards':[],'scope':'Post failed3D GEMM component health only; small allocation/copy/sync; not a hardware stress or model qualification. No reset or node state change.'}
try:
 assert torch.cuda.device_count()==8
 for k in range(8):
  torch.cuda.set_device(k);a=torch.arange(4096,device=k,dtype=torch.float32);b=a.clone();torch.cuda.synchronize(k);assert torch.equal(a,b)and torch.isfinite(b).all();q['cards'].append({'rank':k,'small_copy_sync_passed':True,'free_GiB':torch.cuda.mem_get_info(k)[0]/1024**3});del a,b
 q['completed']=True
finally:(r/'results'/f'kimi-post-shape-health-r3882-{tag}.json').write_text(json.dumps(q,indent=2))
print(json.dumps(q))
PY
