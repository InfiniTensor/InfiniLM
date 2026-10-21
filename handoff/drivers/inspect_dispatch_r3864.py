import pathlib,json,subprocess,datetime
r=pathlib.Path.home()/'hygon-inference-refactor-20261014'
d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':{},'steps':[]}
for n in ['kimi-selective-r3845-audit-start.json','kimi-selective-r3845-audit-queue.json']:
 p=r/n
 if p.exists():d['files'][n]=json.loads(p.read_text())
for n in ['kimi-selective-r3845-audit.json']:
 p=r/'results'/n
 if p.exists():d['files'][n]=json.loads(p.read_text())
d['steps']=[v.split('|')for v in subprocess.check_output(['sacct','-j','910377','-n','-P','-o','JobID,JobName%120,State,ExitCode,Elapsed'],text=True).splitlines()if '3845'in v]
for p in(r/'logs').glob('*3845*'):d['files'][p.name]=p.read_text(errors='replace')[-3000:]
print(json.dumps(d))
