import pathlib,json,subprocess,time,datetime,os
def cmd(args):
 try:return subprocess.check_output(args,text=True,stderr=subprocess.STDOUT,timeout=12).strip()
 except Exception as e:return repr(e)
def read(p):
 try:return pathlib.Path(p).read_text().strip()
 except OSError:return None
def snapshot():
 d={'net':{},'rdma':{}}
 for p in pathlib.Path('/sys/class/net').iterdir():
  d['net'][p.name]={n:int(read(p/'statistics'/n)or 0)for n in['rx_bytes','tx_bytes','rx_packets','tx_packets','rx_errors','tx_errors','rx_dropped','tx_dropped']}
 for dev in pathlib.Path('/sys/class/infiniband').glob('*'):
  for port in(dev/'ports').glob('*'):
   row={n:read(port/n)for n in['state','phys_state','rate','link_layer']};row['counters']={p.name:read(p)for p in(port/'counters').glob('*')if p.name in['port_xmit_data','port_rcv_data','port_xmit_packets','port_rcv_packets','port_xmit_wait','symbol_error','port_rcv_errors','port_xmit_discards','link_downed','link_error_recovery']};d['rdma'][dev.name+'/'+port.name]=row
 return d
a=snapshot();time.sleep(2);b=snapshot();delta={}
for kind in['net','rdma']:
 delta[kind]={}
 for key,v in b[kind].items():
  x=v if kind=='net'else v['counters'];old=a[kind].get(key,{});old=old if kind=='net'else old.get('counters',{});delta[kind][key]={k:int(t)-int(old[k])for k,t in x.items()if str(t).isdigit()and str(old.get(k)).isdigit()}
processes=[]
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  if p.stat().st_uid!=os.getuid():continue
  argv=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if not any(s in argv for s in['modern_kimi_pipeline_r3560.py','modern_kimi_pipeline_r3533.py','modern_glm_fastpath_r3487.py']):continue
  env=dict(v.split('=',1)for v in(p/'environ').read_bytes().decode(errors='replace').split('\0')if'='in v)
  selected={k:v for k,v in env.items()if k.startswith(('NCCL_','UCX_'))or k in['GLOBAL_MODEL_MODE_R2354','KIMI_ENGINE_HEAD']};maps=sorted({line.split()[-1]for line in(p/'maps').read_text().splitlines()if any(s in line for s in['librccl','libinfiniccl','shca-net','libnccl-net','ucx-shca'])})
  processes.append({'pid':int(p.name),'argv':argv[:300],'communication_env':selected,'mapped_communication_libraries':maps})
 except OSError:continue
print(json.dumps({'node':cmd(['hostname','-s']),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sample_seconds':2,'ip_addresses':cmd(['ip','-j','address']),'ip_routes':cmd(['ip','-j','route']),'ibdev2netdev':cmd(['ibdev2netdev']),'rdma_link':cmd(['rdma','link','show']),'before':a,'after':b,'delta':delta,'processes':processes}))
