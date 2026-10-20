import ctypes,datetime,hashlib,json,os,pathlib,subprocess
import torch
torch.set_num_threads(1)
r=pathlib.Path('/refactor');name=os.environ['LATEST_VARIANT_R3508'];assert name in ['Kimi','GLM','Deep'];sdk=r/'install'/f'hygon-{name.lower()}-c3deba4-r3491';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();cfg=r/'results'/f'latest-sdk-r3491-{name}-configure.json';q=json.loads(cfg.read_text());assert torch.cuda.device_count()==8
hardware=[]
for dev in range(8):
 torch.cuda.set_device(dev);x=torch.arange(2048,device=dev,dtype=torch.float32);assert x.sum().item()==2047*2048/2;torch.cuda.synchronize();p=torch.cuda.get_device_properties(dev);assert p.multi_processor_count==80 and p.total_memory>=63*1024**3;hardware.append({'device':dev,'name':p.name,'bytes':p.total_memory,'small_compute_passed':True});del x
torch.cuda.set_device(0)
lib=ctypes.CDLL(str(sdk/'lib/libinfiniops.so'))
assert sha(sdk/'lib/libinfiniccl.so')==q['CCL_sha256']and sha(sdk/'lib/libinfinirt.so')==q['RT_sha256']
maps=sorted({a.split()[-1]for a in pathlib.Path('/proc/self/maps').read_text().splitlines()if'libinfiniops.so'in a});assert len(maps)==1 and pathlib.Path(maps[0]).resolve()==(sdk/'lib/libinfiniops.so').resolve()
out=r/'results'/f'latest-sdk-r3508-{name}-GPU-load.json';assert not out.exists();d={'completed':True,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'variant':name,'step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'pins':q['pins'],'library_sha256':sha(sdk/'lib/libinfiniops.so'),'RT_sha256':q['RT_sha256'],'CCL_sha256':q['CCL_sha256'],'hardware':hardware,'mapped_ops':maps,'source_sha256':sha(r/'incoming/sdk_gpu_load_r3508.py'),'scope':'Fresh bcfe native library loads with GPU visible; all8 small Torch compute. CPU-only no-device abort retained. Actual InfiniOps math/guards/Common/Model/performance not qualified.'};out.write_text(json.dumps(d,indent=2));print(json.dumps(d))
