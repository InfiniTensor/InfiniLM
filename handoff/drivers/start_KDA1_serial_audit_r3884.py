import pathlib,json,subprocess
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'kimi-KDA1-serial-r3884-audit-start.json';assert not out.exists()
with(r/'logs/kimi-KDA1-serial-r3884-audit-controller.log').open('w')as f:p=subprocess.Popen(['python3',str(r/'incoming/queue_KDA1_serial_audit_r3884.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
out.write_text(json.dumps({'submitted':True,'pid':p.pid}));print(out.read_text())
