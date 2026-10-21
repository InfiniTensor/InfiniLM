import pathlib,json,subprocess,datetime,hashlib,re
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
out=r/'followups-r3865-launch.json';assert not out.exists();d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
aa=[v.split('|')for v in subprocess.check_output(['sacct','-j','910381,910374,910377','-n','-P','-o','JobID,JobName%120,State,ExitCode'],text=True).splitlines()]
assert any(v[:4]==['910381.322','glm-PP3-capacity-r3857','COMPLETED','0:0']for v in aa)
rows=[json.loads(p.read_text())for p in(r/'results').glob('glm-PP3-constructor-r3857-910381.322-stage*-rank*.json')]
assert len(rows)==24 and {(q['stage'],q['rank'])for q in rows}=={(s,k)for s in range(3)for k in range(8)}
assert all(q['completed']and q['all24_memory_admitted']and q['all_contexts_closed_without_loading']and q['owner_affinity']['verified']and q['rows']==16 for q in rows)
layers=[];stage_keys=[]
for s,(a,b)in enumerate([(0,26),(26,54),(54,78)]):
 rr=[q for q in rows if q['stage']==s];assert len({q['parameter_keys_sha256']for q in rr})==1
 p=next(q['partition']for q in rr if q['rank']==0);assert [p['begin'],p['end']]==[a,b]
 kk=set(p['keys']);ll={int(v)for k in kk for v in re.findall(r'\.layers\.(\d+)\.',k)};assert ll==set(range(a,b)),(s,ll)
 assert all(not(kk&prev)for prev in stage_keys);stage_keys.append(kk);layers.extend(ll)
assert sorted(layers)==list(range(78))
admit={'completed':True,'constructor_only':True,'rows':16,'cards':24,'all24_owner_close':True,'actual_disjoint_layer_key_coverage':list(range(78)),'minimum_free_GiB':min(q['free_total_after_create'][0]for q in rows)/1024**3,'wholemodel_performance_qualified':False}
(r/'results/glm-PP3-capacity-r3865-audit.json').write_text(json.dumps(admit,indent=2));d['PP3_rows16_audit']=admit
base=json.loads((r/'glm-PP3-capacity-r3857-launch.json').read_text());args=base['args'];args=[v.replace('glm-PP3-capacity-r3857','glm-PP3-capacity-r3865').replace('GLM_CACHE_ROWS_R3627=16','GLM_CACHE_ROWS_R3627=32')for v in args]
with(r/'logs/glm-PP3-capacity-r3865-controller.log').open('w')as f:p=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
d['PP3_rows32']={'pid':p.pid,'args':args,'rows':32,'harness_sha256':base['source_sha256']}
assert any(v[:4]==['910374.370','kimi-root-r3837','COMPLETED','0:0']for v in aa)
assert not(r/'results/kimi-selective-r3845-audit.json').exists()
assert not any('3845'in v[1]and v[2]in['RUNNING','PENDING','CONFIGURING','COMPLETING']for v in aa)
args=['srun','--jobid=910377','--overlap','--exact','-N1','-n1','-c2','-w','COMPUTE_NODE_3181eb8b','--gres=none','--mem=12000M','-t','00:40:00','--job-name=kimi-selective-r3865-audit','--output='+str(r/'logs/kimi-selective-r3865-audit-%J.log'),'python3',str(r/'incoming/audit_kimi_selective_r3845.py')]
with(r/'logs/kimi-selective-r3865-audit-controller.log').open('w')as f:p=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
d['Kimi_audit']={'pid':p.pid,'args':args,'prior_queue_failure':'3845 queue started before producer step existed; original failure retained. Producer terminal0 now checked.'}
out.write_text(json.dumps(d,indent=2));print(json.dumps(d))
