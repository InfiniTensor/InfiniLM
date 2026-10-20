import pathlib,json,subprocess,hashlib,datetime,statistics,collections,math
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'results/parallel-collectives-r3647-audit.json';assert not out.exists();aa=[a.split('|')for a in subprocess.check_output(['sacct','-j','910379','-n','-P','-o','JobID,JobName%120,State,ExitCode,Elapsed'],text=True).splitlines()if a.split('|')[1]=='parallel-collectives-r3638'];assert len(aa)==1 and aa[0][2:4]==['COMPLETED','0:0'];step=aa[0][0];groups=collections.defaultdict(list);raw=0
for node in range(4):
 log=(r/'logs'/f'parallel-collectives-r3638-{step}-{node}.log').read_text();assert'PARALLEL_R3612_NODE_COMPLETE'in log and'SHARED_CACHED_R1780_COMPLETE'in log and'via NET/IBext_v8'in log and not any(x in log for x in ['Traceback','AssertionError','Segmentation fault'])
 masks=[]
 for device in range(8):
  q=json.loads((r/'results'/f'parallel-r3638-{step}-node{node}-gpu{device}.json').read_text());assert q['completed']and q['groups_destroyed']==[32,16,8]and len(q['cases'])==360;masks+=q['CPU_affinity']
  for n,h in q['source_sha256'].items():assert hashlib.sha256((r/'incoming'/n).read_bytes()).hexdigest()==h,n
  for c in q['cases']:
   assert c['finite_exact_oracle_and_input_untouched']and len(c['three_wall_ms_per_call'])==3 and all(math.isfinite(t)and t>0 for t in c['three_wall_ms_per_call']);key=(c['group_size'],(node*8+device)//c['group_size'],c['hidden'],c['batch'],c['dtype'],c['mode']);groups[key].append(statistics.median(c['three_wall_ms_per_call']));raw+=1
 assert len(masks)==len(set(masks))==32
summary=[]
for key,times in sorted(groups.items()):
 size,group,hidden,batch,dtype,mode=key;assert len(times)==size;summary.append({'ranks':size,'group':group,'hidden':hidden,'batch':batch,'dtype':dtype,'mode':mode,'max_rank_median_wall_ms_per_call':max(times)})
d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':True,'producer':aa[0],'normal_GPU_owner_contexts':32,'rank_cases':raw,'group_cases':len(summary),'actual_crossnode_backend':'IBext_v8 in real channel logs onall4nodes','integer_fixture':'rankmod4+1,group totals20/40/80; allintermediatepositiveintegers<=80 exactlyBF16 representable','prior_negative_evidence':'3612 lastcommunicator recreation pluginerror; 3620 rank1..32 intermediateSUM oracle couldround above256 even final528 representable. Both preserved; no rootcause proof or softwarefix inferred solely from newfixture.','scope':'Hardware communication components only, no fullmodelEP/pureTP qualification. All8/16/32communicators keptalive, then normaldestroy. Host wall includes owner/producersync boundaries; not wirebandwidth/networkpeak. F64 only control.','performance':summary};out.write_text(json.dumps(d,indent=2));print(json.dumps({'completed':True,'rank_cases':raw,'group_cases':len(summary)}))
