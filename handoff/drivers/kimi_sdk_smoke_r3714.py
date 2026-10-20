import os,sys
sys.pycache_prefix=os.environ['PYTHONPYCACHEPREFIX']+'/'+str(os.getpid())
import ctypes,json,pathlib,datetime,hashlib
import torch,torch.multiprocessing as mp
r=pathlib.Path('/refactor');base=r/'candidates/kimi-sdk-smoke-5f91ad7-r3714'
def worker(device):
 torch.set_num_threads(1);torch.cuda.set_device(device);allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,set(allowed[4*device:4*device+4]));lib=ctypes.CDLL(str(base/'libkimi_sdk_smoke_r3714.so'));lib.kimi_sdk_smoke_r3714.argtypes=[ctypes.c_int];lib.kimi_sdk_smoke_error_r3714.restype=ctypes.c_char_p;assert lib.kimi_sdk_smoke_r3714(device)==0,lib.kimi_sdk_smoke_error_r3714();q={'completed':True,'device':device,'BF16_IndexSelect_full_output_exact':True,'owner_Runtime_closed':True};(base/f'gpu{device}.json').write_text(json.dumps(q))
if __name__=='__main__':
 mp.spawn(worker,nprocs=8,join=True);records=[json.loads((base/f'gpu{v}.json').read_text())for v in range(8)];p=r/'results/kimi-sdk-smoke-r3714.json';assert not p.exists();p.write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':True,'GPUs':records,'scope':'Actual BF16 IndexSelect firstoperator tinyGPU+owningRuntime/all8normalclose using installedSDKheaders. No checkpointload or fullKimiqualification.','library_sha256':hashlib.sha256((base/'libkimi_sdk_smoke_r3714.so').read_bytes()).hexdigest()},indent=2));print('KIMI_SDK_SMOKE_R3714_COMPLETE',flush=True)
