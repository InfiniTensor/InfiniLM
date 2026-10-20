import pathlib,subprocess,json
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'kimi-selective-r3845-audit-start.json';assert not out.exists()
with(r/'logs/kimi-selective-r3845-audit-controller.log').open('w')as f:p=subprocess.Popen(['python3',str(r/'incoming/queue_kimi_selective_audit_r3845.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
out.write_text(json.dumps({'pid':p.pid,'submitted':True}));print(out.read_text())
