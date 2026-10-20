import pathlib,json,subprocess,datetime
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'glm-capacity-r3721-launch.json';assert not out.exists();old=json.loads((r/'tip-serving-r3698-queue.json').read_text());jobs=[]
for j in old['jobs']:
 if not j['label'].startswith('glm-'):continue
 aa=[a.split('|')for a in subprocess.check_output(['sacct','-j',j['parent'],'-n','-P','-o','JobID,JobName%120,State,ExitCode'],text=True).splitlines()];assert any(a[1]==j['label']and a[2]=='FAILED'for a in aa);assert not any('.'in a[0]and not a[0].endswith(('.batch','.extern'))and a[2]in['RUNNING','PENDING','CONFIGURING','COMPLETING']for a in aa)
 args=[s.replace('r3698','r3721')for s in j['args']];label=j['label'].replace('r3698','r3721')
 with(r/'logs'/(label+'-controller.log')).open('w')as f:p=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 jobs.append({**j,'pid':p.pid,'args':args,'label':label});out.write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'jobs':jobs,'reason':'Previous3698 firstretrievalQA rejected overprovisioned256output ceiling against640context before modelmath; limit now min256,640-input+1. EOS andfulltokenreference stillmandatory; no generation truncation adoption.'},indent=2))
print(json.dumps({'submitted':True,'jobs':[j['label']for j in jobs]}))
