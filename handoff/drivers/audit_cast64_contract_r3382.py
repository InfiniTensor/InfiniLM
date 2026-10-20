import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import collections,concurrent.futures,datetime,functools,hashlib,json,pathlib,subprocess
from fractions import Fraction
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';formats=[(52,11,1023),(23,8,127),(10,5,15),(7,8,127)];width=[8,4,2,2]
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def two(k):return Fraction(1<<k)if k>=0 else Fraction(1,1<<(-k))
def nearest_integer(x):
 q,t=divmod(x.numerator,x.denominator);return q+int(2*t>x.denominator or(2*t==x.denominator and q%2))
@functools.lru_cache(maxsize=400000)
def expected_bits(it,ot,bits):
 p,e,bias=formats[it];op,oe,ob=formats[ot];sign=bits>>(p+e);ex=(bits>>p)&((1<<e)-1);mant=bits&((1<<p)-1);osign=sign<<(op+oe);oinf=((1<<oe)-1)<<op
 if ex==(1<<e)-1:return None if mant else osign|oinf
 if not ex and not mant:return osign
 value=Fraction(mant if not ex else((1<<p)+mant))*two((1-bias-p)if not ex else(ex-bias-p))
 power=value.numerator.bit_length()-value.denominator.bit_length()
 if value<two(power):power-=1
 emin=1-ob;emax=((1<<oe)-2)-ob
 if power<emin:return osign|nearest_integer(value/two(emin-op))
 if power>emax:return osign|oinf
 significand=nearest_integer(value/two(power-op))
 if significand==(1<<(op+1)):significand>>=1;power+=1
 if power>emax:return osign|oinf
 return osign|((power+ob)<<op)|(significand-(1<<op))
def task(payload):
 dev,steps=payload
 rows=[];counts=collections.Counter();mismatches=collections.Counter();examples=[]
 for name in ['Kimi','GLM','Deep']:
  step=steps[name];proof=load(r/'results'/f'cast64-sdk-r3375-{name}-load-build.json');oldq=load(r/'results'/f'cast64-contract-r3376-{name}-{step}-old-gpu{dev}.json');q=load(r/'results'/f'cast64-contract-r3376-{name}-{step}-new-gpu{dev}.json')
  def archive(meta):
   cases=meta['cases'];mode=meta['mode'];assert meta['completed']and meta['Runtime_normally_closed_weak_expired']and meta['wrongowner_untouched']and meta['variant']==name and meta['device']==dev
   assert meta['invalid_checks']==(140 if mode=='new'else 78)and meta['identity_inplace_checks']==(8 if mode=='new'else 6)
   assert meta['library_sha256']==proof['contract_library_sha256'];sdk=load(r/'results'/f'latest-sdk-r2844-{name}-build.json');assert meta['actual_Op_sha256']==(proof['library_sha256']if mode=='new'else sdk['library_sha256'])
   p=r/'results'/f'cast64-contract-r3376-{name}-{step}-{mode}-gpu{dev}.bin';assert sha(p)==meta['archive_sha256']and p.stat().st_size==meta['archive_bytes'];b=p.read_bytes();result={};cursor=0
   for c in cases:
    it,ot,n,shift=[c[k]for k in ['input_type','output_type','count','shift_bytes']];key=it,ot,n,shift;assert key not in result and c['input_readonly_and_guard_untouched'];inp,out=c['input'],c['output'];assert inp['offset']==cursor and inp['bytes']==n*width[it];cursor+=inp['bytes'];assert out['offset']==cursor and out['bytes']==n*width[ot];cursor+=out['bytes']
    aa=b[inp['offset']:inp['offset']+inp['bytes']];bb=b[out['offset']:out['offset']+out['bytes']];assert hashlib.sha256(aa).hexdigest()==inp['sha256']and hashlib.sha256(bb).hexdigest()==out['sha256'];assert c['repeat8_sha256']==[out['sha256']]*8;result[key]=(aa,bb)
   types=range(4)if mode=='new'else range(1,4);assert set(result)=={(a,b,n,s)for a in types for b in types for n in[0,1,17,513,4096]for s in[0,64]}and cursor==len(b)
   return result
  old=archive(oldq);new=archive(q);assert all(new[k]==v for k,v in old.items()),('old9fullrawregression',name,dev)
  counts['old9_regression_cases']+=len(old);counts['new16_cases']+=len(new);counts['Runtime_normal_closures']+=2;counts['invalid_untouched_checks']+=oldq['invalid_checks']+q['invalid_checks']
  for key,(source,out)in new.items():
   it,ot,n,shift=key;p,e,bias=formats[ot];nf=finite=0
   for j in range(n):
    a=int.from_bytes(source[j*width[it]:(j+1)*width[it]],'little');b=int.from_bytes(out[j*width[ot]:(j+1)*width[ot]],'little');gold=expected_bits(it,ot,a)
    if it==ot==0:ok=a==b
    elif gold is None:ok=((b>>p)&((1<<e)-1))==(1<<e)-1 and(b&((1<<p)-1))!=0;nf+=1
    else:ok=b==gold;finite+=1
    if not ok:
     mismatches[name,it,ot]+=1
     if len(examples)<36:examples.append({'variant':name,'device':dev,'input_type':it,'output_type':ot,'count':n,'shift':shift,'index':j,'input_hex':hex(a),'actual_hex':hex(b),'expected_hex':hex(gold)if gold is not None else'NaN_class'})
   counts['CPU_exact_numeric_values']+=finite;counts['CPU_NaN_class_values']+=nf;counts['FP64_identity_raw_values']+=n if it==ot==0 else 0
  rows.append({'variant':name,'device':dev,'old_owner_sha256':sha(r/'results'/f'cast64-contract-r3376-{name}-{step}-old-gpu{dev}.json'),'new_owner_sha256':sha(r/'results'/f'cast64-contract-r3376-{name}-{step}-new-gpu{dev}.json')})
 return{'counts':dict(counts),'mismatches':[{'variant':k[0],'input_type':k[1],'output_type':k[2],'count':v}for k,v in mismatches.items()],'examples':examples,'reports':rows}
TASK_STEPS={}
if __name__=='__main__':
 out=r/'results/cast64-contract-r3382-audit.json';assert not out.exists();d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':False,'qualified':False,'fullmodel_qualified':False}
 try:
  aa=[v.split('|')for v in subprocess.check_output(['sacct','-j','910384,910407','-n','-P','-o','JobID,JobName%110,State,ExitCode,Elapsed,NodeList%160'],text=True).splitlines()if'cast64-contract-r3376-'in v]
  assert len(aa)==3 and all(v[2:4]==['COMPLETED','0:0']for v in aa);TASK_STEPS={v[1].split('-')[-1]:v[0]for v in aa};assert set(TASK_STEPS)=={'Kimi','GLM','Deep'};d['producer_states']=aa
  # Independent rational RNE self checks at representable boundaries and ties.
  assert expected_bits(0,3,0x3ff0100000000000)==0x3f80 and expected_bits(0,3,0x3ff0100000000001)==0x3f81
  assert expected_bits(0,2,0x3ff0020000000000)==0x3c00 and expected_bits(0,1,0x3ff0000010000000)==0x3f800000
  assert expected_bits(0,2,0x3e60000000000000)==0 and expected_bits(0,2,0x3e60000000000001)==1
  for name in TASK_STEPS:
   proof=load(r/'results'/f'cast64-sdk-r3375-{name}-load-build.json');assert proof['completed']and proof['strict_public_headers_unchanged_count']>=633
   for n,h in proof['source_sha256'].items():assert sha(r/'incoming'/n)==h
   for mode in ['old','new']:
    q=load(r/'results'/f'cast64-contract-r3376-{name}-{TASK_STEPS[name]}-{mode}.json');assert q['completed']and len(q['owners'])==8
    for o in q['owners']:assert sha(r/'results'/o['file'])==o['sha256']
  with concurrent.futures.ProcessPoolExecutor(max_workers=8)as pool:values=list(pool.map(task,[(dev,TASK_STEPS)for dev in range(8)]))
  counts=collections.Counter();mismatches=collections.Counter();examples=[];reports=[]
  for v in values:
   counts.update(v['counts']);reports+=v['reports'];examples+=v['examples']
   for a in v['mismatches']:mismatches[a['variant'],a['input_type'],a['output_type']]+=a['count']
  assert counts['new16_cases']==3840 and counts['old9_regression_cases']==2160 and counts['Runtime_normal_closures']==48
  for v in aa:
   log=(r/'logs'/f'{v[1]}-{v[0]}.log').read_text(errors='replace');assert log.count('CAST64_CONTRACT_R3376_ALL8_CLOSED')==2 and'SHARED_CACHED_R1780_COMPLETE'in log and not any(w in log for w in ['Traceback','AssertionError','Segmentation fault'])
  d.update(completed=True,qualified=not mismatches,counts=dict(counts),mismatches=[{'variant':k[0],'input_type':k[1],'output_type':k[2],'count':v}for k,v in sorted(mismatches.items())],examples=examples[:72],all_old9_fullraw_regressions_exact=True,new_FP64_related_pairs_qualified=all(k[1]!=0 and k[2]!=0 for k in mismatches),reports=reports,scope='ThreeisolatedSDKs,8owners each,all16pairs exactrationalCPU_RNE includingnormal/subnormal/underflow/overflow/ties/signedzero/Inf; NaNclass exceptF64identityrawbits. Old9types byteexactregression in separateprocesses;8ownrepeats/guards/invalid/empty/normalweakRuntimeclose. Finiteerrorsretained; noCCL/wholemodel/TPS/capture qualification.')
 except BaseException as e:
  import traceback
  d.update(error=repr(e),traceback=traceback.format_exc());out.write_text(json.dumps(d,indent=2));raise
 out.write_text(json.dumps(d,indent=2));print('CAST64_CONTRACT_R3382_CPU_AUDIT_COMPLETE',d['qualified'],dict(counts),flush=True)
