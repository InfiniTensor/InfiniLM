import ctypes,hashlib,json,time,os
import numpy as np
import torch

def run(g):
 r,tag,lib,ctx,cfg,forward,check,report,path=[g[k] for k in ['r','tag','lib','ctx','cfg','forward','check','report','path']]
 policy=lib.modern_deep_precision_policy_r3443;policy.argtypes=[ctypes.c_int]
 stats=lib.modern_deep_precision_stats_r3443;stats.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t]
 def counts():
  rows=[]
  for dev in range(8):
   a=(ctypes.c_uint64*6)();assert stats(dev,a,6)==0;rows.append(list(a))
  return rows
 def save():path.write_text(json.dumps(report,indent=2))
 assert policy(-1)==policy(5)==policy(99)==1 and policy(0)==0;assert counts()==[[0]*6 for _ in range(8)]
 proof=json.loads((r/'results/deep-f64-r3449-audit.json').read_text());assert proof['completed'] and proof['F64_all_cases_CPU_FP64sum_RNE_F32_BF16_and_rows_repeat_exact']
 report['source_sha256']={n:hashlib.sha256((r/'incoming'/n).read_bytes()).hexdigest() for n in ['deep_wide_protocol_r3457.py','modern_deep_wide_r3457.py','deep_wide_r3457_inside.sh']}
 g['run_chat']('before');assert counts()==[[0]*6 for _ in range(8)]
 natural=json.loads((r/'results'/f'deep-serving-cases-r2130-{tag}.json').read_text())['cases'];short=[{'label':'short_'+v['label'],'input_ids':v['input_ids']} for v in g['chat_payload']['cases']];long=[{'label':'long_'+v['label'],'input_ids':v['input_ids']} for v in natural]
 category=os.environ['DEEP_WIDE_CASE_FILTER_R3457'];assert category in ['short','long'];cases=short if category=='short' else long
 report.update(case_filter=category,precision_diagnostics=[],B1_policy_comparisons=[],scope='Latest3e0 Deep full43 diagnostic, mode0 native no extra sync; mode1 native plus matched sync;2attentionF64;3addsFFN;4addsindexer if path used. B1 forcednative. Exactsame-input duplicate/repeat/raw preserved; not B2 quality or TPS qualification.')
 for case in cases:
  prompt=case['input_ids'];P=len(prompt);label=case['label'];references={};singles=[]
  for mode in range(5):
   assert policy(mode)==0;check(lib.modern_native_engine_reset(ctx));before=counts();code,x,ids=forward(prompt,list(range(P)),[0],[0],[P],[0,P]);check(code);check(lib.modern_native_engine_rank_logits_equal(ctx));assert counts()==before
   singles.append(x.copy());f=r/'results'/f'deep-wide-r3457-{tag}-{label}-B1-policy{mode}.bin';assert not f.exists();f.write_bytes(x.tobytes());v=torch.from_numpy(x).view(torch.bfloat16);assert torch.isfinite(v).all() and int(v.argmax(-1))==int(ids[0]);report['B1_policy_comparisons'].append({'label':label,'policy':mode,'file':f.name,'sha256':hashlib.sha256(x.tobytes()).hexdigest(),'id':int(ids[0]),'counters':counts()});save()
  assert all(np.array_equal(singles[0],v) for v in singles[1:]),('B1 math must remain exact',label)
  for swap in [False,True]:
   rows=[1,0] if swap else [0,1]
   for epoch,mode in enumerate([0,1,2,3,4,4,3,2,1,0]):
    assert policy(mode)==0;check(lib.modern_native_engine_reset(ctx));current=None
    for step in range(4):
     flat=prompt*2 if step==0 else [int(v) for v in current];past=0 if step==0 else P+step-1;width=P if step==0 else 1;before=counts();tick=time.perf_counter();code,x,ids=forward(flat,list(range(past,past+width))*2,rows,[past,past],[past+width,past+width],[0,width,2*width]);check(code);elapsed=time.perf_counter()-tick;after=counts();delta=[[b-a for a,b in zip(v,w)] for v,w in zip(before,after)]
     for v in delta:
      assert v[1]==(43 if mode>=2 else 0) and v[2]==(43 if mode>=3 else 0) and (mode>=4 or v[3]==0),('dispatch_counts',label,mode,step,v)
      if mode==0:assert v==[0]*6
      else:assert v[0]>=86
     check(lib.modern_native_engine_rank_logits_equal(ctx));v=torch.from_numpy(x).view(torch.bfloat16);assert torch.isfinite(v).all() and v.argmax(-1).tolist()==ids.tolist();raw=x.tobytes();digest=hashlib.sha256(raw).hexdigest();file=r/'results'/f'deep-wide-r3457-{tag}-{label}-swap{int(swap)}-epoch{epoch}-mode{mode}-step{step}.bin';assert not file.exists();file.write_bytes(raw);key=mode,swap,step;gold=references.setdefault(key,digest)
     report['precision_diagnostics'].append({'label':label,'epoch':epoch,'policy':mode,'cache_rows':rows,'step':step,'input_ids':flat,'past':past,'selected_ids':ids.tolist(),'duplicate_fullraw_exact':bool(np.array_equal(x[0],x[1])),'samegeometry_repeat_exact':digest==gold,'all8rank_fullraw_equal':True,'file':file.name,'sha256':digest,'counter_delta':delta,'host_seconds_diagnostic':elapsed,'B1_prefill_raw_match_each_row':[bool(np.array_equal(z,singles[0][0])) for z in x] if step==0 else None});report['phase']={'case':label,'swap':swap,'epoch':epoch,'mode':mode,'step':step};save();current=ids
 assert policy(4)==0;before=counts();g['run_chat']('after');assert counts()==before
 report['policy_summary']={str(mode):{'cases':len(v:=[x for x in report['precision_diagnostics'] if x['policy']==mode]),'duplicate_raw_exact':sum(x['duplicate_fullraw_exact'] for x in v),'samegeometry_repeat_exact':sum(x['samegeometry_repeat_exact'] for x in v)} for mode in range(5)}
 report.update(B2_qualified=False,wholemodel_B2_useful_throughput_measured=False,all_counters=counts());save()
