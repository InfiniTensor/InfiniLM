import pathlib,json,subprocess,datetime
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'proofs':{},'steps':[],'constructors':[],'replay':[],'logs':{}}
for n in ['kimi-selective-r3845-audit.json','glm-PP3-capacity-r3865-audit.json','kimi-projection-r3866-build.json']:
 p=r/'results'/n
 if p.exists():d['proofs'][n]=json.loads(p.read_text())
d['steps']=[v.split('|')for v in subprocess.check_output(['sacct','-j','910373,910377,910381','-n','-P','-o','JobID,JobName%120,State,ExitCode,Elapsed'],text=True).splitlines()if any(k in v.split('|')[1]for k in['3865','3866'])]
for p in(r/'results').glob('glm-PP3-constructor-r3857-*.json'):
 q=json.loads(p.read_text());q.pop('scope',None)
 if 'partition'in q:q['partition'].pop('keys',None)
 d['constructors'].append(q)
for p in(r/'results').glob('kimi-KDA1-replay-r3866-*-rank*.json'):d['replay'].append(json.loads(p.read_text()))
for a in d['steps']:
 for p in(r/'logs').glob(a[1]+'-'+a[0]+'*.log'):d['logs'][p.name]=p.read_text(errors='replace')[-4500:]
print(json.dumps(d))
