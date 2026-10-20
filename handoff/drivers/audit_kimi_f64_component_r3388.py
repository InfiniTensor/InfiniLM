import datetime,hashlib,json,pathlib,subprocess,traceback
import numpy as np
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'results/kimi-f64-r3388-audit.json';assert not out.exists();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':False,'scope':'IndependentCPU actualfullinputpartials/outputs on8ranks; FP64sum ofboundedrangeFP32 inputs thenRNEFP32/BF16, comparesnativeF32vsactualSDKCastF64+CCL. Syntheticcases demonstrateconditionalnumericalbehavior only, notremainingKimi modelrootcause/fix/TPS orvendorfault.'}
def read(v):
 p=r/'results'/v['file'];assert p.parent==r/'results';b=p.read_bytes();assert len(b)==v['bytes']and hashlib.sha256(b).hexdigest()==v['sha256'];dt=np.float32 if v['dtype']=='torch.float32'else np.uint16;x=np.frombuffer(b,dt).reshape(v['shape']);return b,x
def bf16(a):
 b=a.astype(np.float32).view(np.uint32);return((b+np.uint32(0x7fff)+((b>>16)&1))>>16).astype(np.uint16)
try:
 aa=[a.split('|')for a in subprocess.check_output(['sacct','-j','910377','-n','-P','-o','JobID,JobName%110,State,ExitCode,Elapsed'],text=True).splitlines()];found=[a for a in aa if a[1]=='kimi-f64-r3385-component'];assert len(found)==1 and found[0][2:4]==['COMPLETED','0:0'];step=found[0][0];d['producer']=found[0];build=json.loads((r/'results/kimi-f64-r3385-build.json').read_text());assert build['completed']and sha(r/'candidates/kimi-f64-3e0d89fe-r3385/libkimi_f64_r3385.so')==build['library_sha256']
 for n,h in build['source_sha256'].items():assert sha(r/'incoming'/n)==h
 for n,h in build['mapped_libraries'].items():assert sha(pathlib.Path(n.replace('/refactor',str(r),1)))==h
 groups={};affinity=[];reports=[];files=0
 for rank in range(8):
  p=r/'results'/f'kimi-f64-r3385-{step}-rank{rank}.json';q=json.loads(p.read_text());assert q['completed']and q['numeric_collection_before_close']and q['communicator_normally_closed']and not q.get('error')and len(q['cases'])==24;assert q['library_sha256']==build['library_sha256']and q['source_sha256']==build['source_sha256']['probe_kimi_f64_component_r3385.py'];assert len(q['CPU_affinity'])==4 and not set(q['CPU_affinity'])&set(affinity);affinity+=q['CPU_affinity']
  for c in q['cases']:
   assert c['guards_invalid_wrongowner_readonly_passed'];raw,x=read(c['input']);files+=1;assert np.isfinite(x).all()and all(np.array_equal(row,x[0])for row in x);assert c['N']==7168 and c['M']in[4,8,16];assert len(c['outputs'])==4 and[v['mode']for v in c['outputs']]==[0,1,1,0]
   for e in c['outputs']:
    b,y=read(e['raw']);bb,yb=read(e['BF16_raw']);files+=2;assert np.isfinite(y).all()and y.shape==(c['M'],7168);assert np.array_equal(yb,bf16(y));assert len(e['F32_sha256_repeat8'])==len(e['BF16_sha256_repeat8'])==8;assert e['F32_sha256_repeat8'][-1]==hashlib.sha256(b).hexdigest()and e['BF16_sha256_repeat8'][-1]==hashlib.sha256(bb).hexdigest();assert e['own_repeat8_exact']==(len(set(e['F32_sha256_repeat8']))==1);assert e['duplicate_rows_fullraw_exact']==all(np.array_equal(row,y[0])for row in y)
   groups.setdefault((c['M'],c['fixture'],c['shift_bytes'],c['inplace']),[]).append({'rank':rank,'input':x,'case':c})
  reports.append({'rank':rank,'report_sha256':sha(p),'normal_communicator_closed':True})
 assert len(groups)==24;summary=[];candidate_pass=True
 for key,rows in sorted(groups.items()):
  assert len(rows)==8;gold=np.sum([v['input'].astype(np.float64)for v in rows],axis=0).astype(np.float32);gold_bf=bf16(gold);goldbytes=gold.tobytes();gold_bfbytes=gold_bf.tobytes();modes={}
  for mode in[0,1]:
   results=[]
   for v in rows:
    for e in v['case']['outputs']:
     if e['mode']!=mode:continue
     b,y=read(e['raw']);bb,yb=read(e['BF16_raw']);results.append({'rank':v['rank'],'F32_sha256':hashlib.sha256(b).hexdigest(),'BF16_sha256':hashlib.sha256(bb).hexdigest(),'F32_CPU_reference_exact':b==goldbytes,'BF16_CPU_reference_exact':bb==gold_bfbytes,'F32_changed_words':int(np.count_nonzero(y.view(np.uint32)!=gold.view(np.uint32))),'BF16_changed_words':int(np.count_nonzero(yb!=gold_bf)),'duplicate_rows_fullraw_exact':e['duplicate_rows_fullraw_exact'],'BF16_duplicate_rows_exact':all(np.array_equal(row,yb[0])for row in yb),'own_repeat8_exact':e['own_repeat8_exact']})
     if mode:assert e['F32_sha256_repeat8']==[hashlib.sha256(goldbytes).hexdigest()]*8 and e['BF16_sha256_repeat8']==[hashlib.sha256(gold_bfbytes).hexdigest()]*8
   assert len(results)==16
   passed=all(v['F32_CPU_reference_exact']and v['BF16_CPU_reference_exact']and v['duplicate_rows_fullraw_exact']and v['own_repeat8_exact']for v in results)
   if mode:candidate_pass&=passed
   modes[str(mode)]={'all_exact_and_repeat':passed,'F32_distinct_output_hashes_across_rank_epoch':len({v['F32_sha256']for v in results}),'BF16_distinct_output_hashes_across_rank_epoch':len({v['BF16_sha256']for v in results}),'row_equal_records':sum(v['duplicate_rows_fullraw_exact']for v in results),'BF16_row_equal_records':sum(v['BF16_duplicate_rows_exact']for v in results),'max_F32_changed_words':max(v['F32_changed_words']for v in results),'max_BF16_changed_words':max(v['BF16_changed_words']for v in results)}
  summary.append({'M':key[0],'fixture':key[1],'shift_bytes':key[2],'inplace':key[3],'modes':modes})
 assert candidate_pass
 d.update(completed=True,cases=192,normal_communicators_closed=8,raw_files=files,summary=summary,reports=reports,F64_all_cases_CPU_FP64sum_RNE_F32_BF16_and_rows_repeat_exact=True,F32_cases_with_row_mismatch=sum(v['modes']['0']['row_equal_records']<16 for v in summary),F32_cases_with_BF16_visible_row_mismatch=sum(v['modes']['0']['BF16_row_equal_records']<16 for v in summary),scope=d['scope'])
except BaseException as e:d.update(error=repr(e),traceback=traceback.format_exc());out.write_text(json.dumps(d,indent=2));raise
out.write_text(json.dumps(d,indent=2));print('KIMI_F64_R3388_AUDIT_COMPLETE',d['F32_cases_with_row_mismatch'],d['F32_cases_with_BF16_visible_row_mismatch'],flush=True)
