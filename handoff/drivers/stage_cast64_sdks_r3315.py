import datetime,difflib,hashlib,json,pathlib,shutil
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';base=r/'candidates/cast64-3e0d89fe-r3315';out=r/'results/cast64-sdks-r3315-stage.json';assert not base.exists()and not out.exists();base.mkdir();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'workspace_root':str(r),'ready':False,'variants':{},'scope':'Source-onlynewOpscopies overcurrentc86, all3model-specificpatchsets preserved. OnlyHygonCast implementationaddsF64pairs withdirectintegerIEEE RNE/bitidentity, existing9pairGPUbody unchanged; captureSafeFalse anddevice/shape/stride/overlapguards retained. NewSDKprefixes/freshbuilds; pinnedRTb413/CCL6f binaries reusedSHAidentical. NoexistingSDK/Common/model touched, noGPU/modelqualificationyet.'}
for name in['Kimi','GLM','Deep']:
 parent=r/'candidates/rebased-3e0d89fe-r2842/Ops'/name/'InfiniOps';target=base/name/'InfiniOps';shutil.copytree(parent,target,ignore=shutil.ignore_patterns('.git','generated','build','_deps','__pycache__','*.so','*.o','*.a'))
 p=target/'src/native/cuda/hygon/ops/cast/kernel.h';old=p.read_text();assert 't==DataType::kFloat64'not in old;new=(r/'incoming/hygon_cast64_kernel_r3314.h').read_text();p.write_text(new)
 proof=json.loads((r/'results'/f'latest-sdk-r2844-{name}-build.json').read_text());assert proof['completed']and proof['pins']['InfiniOps']=='c86b4f229e5a61331e281964df8b42f100fdd178';current=r/'install'/f'hygon-{name.lower()}-3e0d89fe-r2844';assert sha(current/'lib/libinfiniops.so')==proof['library_sha256'];changed=[];files={}
 for a in(parent/'src').rglob('*'):
  if not a.is_file():continue
  n=a.relative_to(parent).as_posix();b=target/n;assert b.exists();files[n]=sha(b)
  if sha(a)!=files[n]:changed.append(n)
 assert changed==['src/native/cuda/hygon/ops/cast/kernel.h'],changed
 q=json.loads((r/'results'/f'latest-sdk-r2844-{name}-configure.json').read_text());prefix=r/'install'/f'hygon-{name.lower()}-cast64-r3315';build=r/'build'/f'ops-{name.lower()}-cast64-r3315';assert not prefix.exists()and not build.exists()
 v={'source':str(target),'build':str(build),'install':str(prefix),'parent_install':str(current),'pins':proof['pins'],'parent_Op_sha256':proof['library_sha256'],'RT_sha256':proof['RT_sha256'],'CCL_sha256':proof['CCL_sha256'],'configure':q['configuration'],'changed':changed,'cast_before_sha256':sha(parent/'src/native/cuda/hygon/ops/cast/kernel.h'),'cast_after_sha256':sha(p),'diff':''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True))),'source_sha256':files};d['variants'][name]=v
d['ready']=True;out.write_text(json.dumps(d,indent=2));print(json.dumps({'ready':True,'variants':list(d['variants']),'changed_per_variant':1}))
