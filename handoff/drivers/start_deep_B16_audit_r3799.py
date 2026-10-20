import pathlib,json,subprocess
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'deep-B16-r3799-audit-start.json';assert not out.exists()
with(r/'logs/deep-B16-r3799-audit-controller.log').open('w')as f:p=subprocess.Popen(['python3',str(r/'incoming/queue_deep_B16_audit_r3799.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
out.write_text(json.dumps({'started':True,'pid':p.pid},indent=2));print(json.dumps({'started':True,'pid':p.pid}))
