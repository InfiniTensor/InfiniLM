import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import pathlib,json,hashlib,subprocess,datetime,math,traceback
import numpy as np
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'results/kimi-root-r3818-audit.json';assert not out.exists();d={'completed':False,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'main_goal_metric':False,'speculative_performance_qualified':False};sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
try:
 launch=json.loads((r/'kimi-root-r3810-launch.json').read_text());aa=[v.split('|')for v in subprocess.check_output(['sacct','-j',launch['parent'],'-n','-P','-o','JobID,JobName%120,State,ExitCode,Elapsed'],text=True).splitlines()if v.split('|')[1]==launch['label']];assert len(aa)==1 and aa[0][2:4]==['COMPLETED','0:0'],aa;tag=launch['run'];reports=[];rawchecked=0;proof=json.loads((r/'results/kimi-spec-alias-r3775-build.json').read_text());assert proof['completed']and sha(r/'candidates/kimi-spec-alias-5f91ad7-r3775/libmodern_kimi_spec_alias_r3775.so')==proof['library_sha256']
 for n,h in launch['source_sha256'].items():assert sha(r/'incoming'/n)==h
 for node in range(4):
  p=r/'results'/f'kimi-spec-r3810-{tag}-node{node}.json';q=json.loads(p.read_text());assert q['completed']and q['all8_owned_workers_shutdown_completed']and q['owner_affinity']['verified']and q['root_collection_completed']and q['root_trace_only']and not q['main_goal_metric']and not q['performance_qualified'];assert q['library_sha256']==proof['library_sha256']and q['harness_files_sha256']==launch['source_sha256'];assert len(q['QA'])==16 and all(v['complete_frozen_IDs_exact']for v in q['QA']);assert q['root_plan']==launch['selected_plan']
  expected=q['root_plan']['expected_AR_ids'];cases=q['spec_cases'];assert next(v for v in cases if v['label']=='root-AR-before')['generated_ids']==[expected]==next(v for v in cases if v['label']=='root-AR-after')['generated_ids']
  paths=q['root_paths'];assert[v['label']for v in paths]==['serial-traced','block-traced','block-untraced-1','block-untraced-2'];assert paths[0]['generated_ids']==expected and paths[1]['generated_ids']==paths[2]['generated_ids']==paths[3]['generated_ids'];assert q['root_traced_vs_untraced_IDs_exact']and q['root_untraced_repeats_exact']
  log=(r/'logs'/f'{launch["label"]}-{aa[0][0]}-{node}.log').read_text(errors='replace');assert 'KIMI_SPEC_R3634_COMPLETE'in log and 'SHARED_CACHED_R1780_COMPLETE'in log and not any(v in log for v in ['Traceback','AssertionError','Segmentation fault'])
  for v in q['full_logit_records']:
   assert v['finite']and v['CPUargmax_equals_GPU']
   if node!=3 or not v['raw_persisted']:continue
   pp=r/'results'/v['file'];b=pp.read_bytes();assert pp.name==v['file']and len(b)==math.prod(v['shape'])*2 and hashlib.sha256(b).hexdigest()==v['sha256'];x=(np.frombuffer(b,np.uint16).astype(np.uint32)<<16).view(np.float32).reshape(v['shape']);assert np.isfinite(x).all()and x.argmax(-1).tolist()==v['outputs'];rawchecked+=1
  reports.append(q)
 assert all(v['root_paths']==reports[0]['root_paths']for v in reports[1:]);virtualroot=pathlib.Path(reports[0]['trace_root']);assert virtualroot.parent==pathlib.Path('/refactor/results');root=r/'results'/virtualroot.name;journals=[];comparisons=[];totalbytes=0
 points=['entry','attn_res','attn_norm','attn_output','attn_added','ffn_res','ffn_norm','ffn_output','exit']
 def read_trace(label,call):
  nonlocal_dummy=None;directory=root/label/call;result={}
  for p in sorted(directory.glob('stage*-rank*.jsonl')):
   if '-kda-call'in p.name:continue
   rows=[json.loads(v)for v in p.read_text().splitlines()];assert rows
   journals.append({'file':p.relative_to(r).as_posix(),'sha256':sha(p),'entries':len(rows)})
   for row in rows:
    if row['point']not in points:continue
    key=(row['stage'],row['rank'],row['layer'],row['point']);assert key not in result;result[key]=(row,directory)
  return result
 serial1=read_trace('serial-traced','serial-first');serial2=read_trace('serial-traced','serial-second');block=read_trace('block-traced','block');assert set(serial1)==set(serial2)==set(block)
 assert{key[2]for key in block}==set(range(93))
 for key in sorted(block,key=lambda k:(k[2],points.index(k[3]),k[1])):
  stage,rank,layer,point=key
  if rank not in[0,6]:continue
  mats=[];hashes=[]
  for mapping in[serial1,serial2,block]:
   row,directory=mapping[key];p=directory/f'stage{stage}-rank{rank}-layer{layer}-{point}-full-bf16.bin';b=p.read_bytes();assert len(b)==row['tokens']*row['hidden']*2;totalbytes+=len(b);x=np.frombuffer(b,np.uint16).reshape(row['tokens'],row['hidden']);f=(x.astype(np.uint32)<<16).view(np.float32);assert np.isfinite(f).all();mats.append((x,f));hashes.append(hashlib.sha256(b).hexdigest())
  assert mats[0][0].shape==mats[1][0].shape==(1,block[key][0]['hidden'])and mats[2][0].shape==(2,block[key][0]['hidden'])
  for token in[0,1]:
   ref,ff=mats[token];actual,af=mats[2];neq=int(np.count_nonzero(ref[0]!=actual[token]));comparisons.append({'stage':stage,'rank':rank,'layer':layer,'point':point,'token_offset':token,'diagnostic_BF16_different_elements':neq,'max_absolute_difference':float(np.abs(ff[0]-af[token]).max()),'serial_sha256':hashes[token],'block_sha256':hashes[2]})
 first={}
 for v in comparisons:
  if v['diagnostic_BF16_different_elements']:
   first.setdefault(str(v['token_offset']),v)
 d.update(completed=True,producer=aa[0],normal_owned_workers_closed=32,CPU_fullvocab_raw_files=rawchecked,full_known_QA=16,serial_full_frozen_IDs_exact=True,traced_and_untraced_block_full_IDs_exact=True,block_complete_IDs_equal_serial=reports[0]['root_paths'][1]['generated_ids']==reports[0]['root_paths'][0]['generated_ids'],root_plan=launch['selected_plan'],root_paths=reports[0]['root_paths'],generic_trace_CPU_checked_bytes=totalbytes,generic_trace_journals=journals,first_generic_diagnostic_difference_by_token=first,diagnostic_comparisons=comparisons,scope='First actual acceptedblock fromidenticalprefix, teacher input only forcorrectness localization. Complete serial frozenAR, traced/untraced fullcontinuation exact, finiteCPUargmax/owner/Slurm audited. Rawdifferences diagnostic only, nottokenfailure or performance. Genericfulltrace rank0/6 all93layers; otherTP ranks producerlastsnapshots andselectedKDA details persisted butnotindependentlycompared here. No draftperformance gain or mainmetric adoption.')
except BaseException as e:d.update(error=repr(e),traceback=traceback.format_exc());out.write_text(json.dumps(d,indent=2));raise
out.write_text(json.dumps(d,indent=2));print('KIMI_ROOT_R3818_AUDITED',d['block_complete_IDs_equal_serial'],first)
