"""Bounded node-local cache; no GPU allocation or original checkpoint changes."""
import collections,datetime,hashlib,json,os,pathlib,re,signal,socket,struct,subprocess,sys,time,traceback
r=pathlib.Path('/refactor');name=os.environ['CACHE_MODEL_R3604'];node=socket.gethostname();stage=int(os.environ.get('CACHE_STAGE_R3604','0'));pp=int(os.environ.get('CACHE_PP_R3604','1'));tag=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];base=pathlib.Path('/dev/shm')/('iloader-r3607-'+str(os.getuid()));base.mkdir(mode=0o700,exist_ok=True);cache_name=name.lower()+'-pp'+str(pp)+'-s'+str(stage);model={'GLM':pathlib.Path('/old/models/ZhipuAI-GLM-5.3'),'Kimi':pathlib.Path('/shared/Kimi-K3'),'Deep':pathlib.Path('/old/models/hygon-DeepSeek-V4-Flash-Channel-INT8-w8a8')}[name]
# The existing metadata is authoritative for the exact deployed checkpoint path.
meta=json.loads((r/'results/model-linear-metadata-r3293.json').read_text())['models'][name];host=pathlib.Path(meta['path']);home=pathlib.Path('/path/to/cluster-home');model=pathlib.Path(str(host).replace(str(home/'hygon-inference-20261013'),'/old').replace(str(home/'shared'),'/shared'));assert model.is_dir(),model
path=r/'results'/f'weight-cache-r3607-{tag}-{node}-{name}.json';assert not path.exists();d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'node':node,'model':name,'source':str(model),'PP':pp,'stage':stage,'step':tag,'completed':False,'cache_ready':False,'loader_commit':'88d6bac6a80a424201da1688989d1d87e3217a93','state':'planning','GPU_work_performed':False,'scope':'Selected immutable full checkpoint files cached under128GiB, all other paths read original files through explicit overlay. One daemon per host/model stage; no GPU tensor format or TP packing changes. Full engine-token/performance qualification separate.'}
def save():
 t=path.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(path)
def call(args,timeout=60):
 p=subprocess.run([sys.executable,'-m','infiniloader',*args],capture_output=True,text=True,timeout=timeout)
 if p.returncode:raise RuntimeError({'args':args,'code':p.returncode,'out':p.stdout[-1500:],'err':p.stderr[-3000:]})
 return p
def mem():
 from infiniloader.memory import memory_info
 return memory_info()
def relevant(key):
 if name=='Kimi':
  match=re.search(r'language_model\.model\.layers\.(\d+)\.',key);bounds={4:[0,24,47,71,93],2:[0,46,93],1:[0,93]}[pp]
  return bounds[stage]<=int(match[1])<bounds[stage+1]if match else(key=='language_model.model.embed_tokens.weight'and stage==0 or key.startswith(('language_model.lm_head.','language_model.model.norm.','language_model.model.output_attn_res_'))and stage==pp-1)
 if name=='GLM':
  match=re.search(r'model\.layers\.(\d+)\.',key);bounds={4:[0,18,38,58,78],2:[0,38,78],1:[0,78]}[pp]
  return bounds[stage]<=int(match[1])<bounds[stage+1]if match else(key=='model.embed_tokens.weight'and stage==0 or key.startswith(('lm_head.','model.norm.'))and stage==pp-1)
 return not key.startswith(('mtp.','vision.'))
save();stop=False
def stopped(*args):
 global stop;stop=True
signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
try:
 d['doctor']=json.loads(call(['doctor','--base-dir',str(base),'--json']).stdout);d['memory_before']=mem();mapping=json.loads((model/'model.safetensors.index.json').read_text())['weight_map'];wanted=collections.defaultdict(list)
 for k,f in mapping.items():
  if relevant(k):wanted[f].append(k)
 rows=[]
 for filename,keys in wanted.items():
  with(model/filename).open('rb')as f:n=struct.unpack('<Q',f.read(8))[0];assert 0<n<150000000;header=json.loads(f.read(n))
  missing=[k for k in keys if k not in header];d.setdefault('index_names_absent_from_that_file',{})[filename]=missing
  actual=[k for k in header if k!='__metadata__'and relevant(k)];useful=sum(header[k]['data_offsets'][1]-header[k]['data_offsets'][0]for k in actual);total=(model/filename).stat().st_size;rows.append({'file':filename,'bytes':total,'required_tensor_bytes':useful,'required_keys':len(actual),'efficiency':useful/total})
 cap=128*1024**3;selected=[];used=0
 for row in sorted(rows,key=lambda v:(-v['efficiency'],v['file'])):
  if used+row['bytes']<=cap:selected.append(row);used+=row['bytes']
 assert selected and used<=cap;d.update(selected=selected,selected_file_bytes=used,required_files=len(rows),required_source_tensor_bytes=sum(v['required_tensor_bytes']for v in rows),cached_required_tensor_bytes=sum(v['required_tensor_bytes']for v in selected),cache_name=cache_name,base_dir=str(base),state='preloading');save()
 args=['start',str(model),'--base-dir',str(base),'--name',cache_name,'--workers','1','--chunk-size','16777216','--max-memory-fraction','0.95','--max-memory-bytes','132GiB','--max-resident-bytes','132GiB','--reserve-bytes','48GiB','--wait-timeout','14400']
 for row in selected:args+=['--pattern',row['file']]
 begin=time.perf_counter();p=call(args,14500);view=pathlib.Path(p.stdout.strip().splitlines()[-1]);assert view.is_dir();d.update(preload_seconds=time.perf_counter()-begin,view=str(view),preload_stderr_tail=p.stderr[-1500:]);status=json.loads(call(['status',cache_name,'--base-dir',str(base),'--json']).stdout);d['status']=status;assert status['ready']if'ready'in status else status.get('state','ready')=='ready'
 overlay=base/('overlay-'+cache_name);overlay.mkdir(mode=0o700,exist_ok=False);cached={v['file']for v in selected}
 for p in model.iterdir():
  if p.is_file():(overlay/p.name).symlink_to(view/p.name if p.name in cached else p)
 d['overlay']=str(overlay);samples=[]
 for row in selected:
  rel=row['file'];size=row['bytes'];offsets=sorted(set([0,max(0,size//2-4096),max(0,size-8192)]));blocks=[]
  with(model/rel).open('rb')as src,(overlay/rel).open('rb')as got:
   for offset in offsets:
    src.seek(offset);got.seek(offset);a=src.read(8192);b=got.read(8192);assert a==b;blocks.append({'offset':offset,'bytes':len(a),'sha256':hashlib.sha256(b).hexdigest()})
  samples.append({'file':rel,'blocks':blocks})
 d.update(sampled_source_vs_memfd_exact=samples,cache_ready=True,state='ready',memory_after=mem());save();print('WEIGHT_CACHE_R3604_READY',node,name,used,d['preload_seconds'],str(overlay),flush=True)
 # The owning Slurm step remains alive; no daemon is leaked beyond this step.
 deadline=time.monotonic()+20*3600
 while not stop and time.monotonic()<deadline:
  d['heartbeat_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();d['memory_live']=mem();save();time.sleep(30)
 d.update(state='stopping');save();call(['stop',cache_name,'--base-dir',str(base)],60);d.update(completed=True,state='stopped');save()
except BaseException as e:
 d.update(error=repr(e),traceback=traceback.format_exc(),state='failed');save();raise
