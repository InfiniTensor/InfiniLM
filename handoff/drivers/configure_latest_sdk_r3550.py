import datetime,hashlib,json,pathlib,shutil,subprocess,sys
r=pathlib.Path('/refactor');name=sys.argv[1];assert name in ['Kimi','GLM','Deep'];base=r/'candidates/rebased-06d0c03-r3548/Ops'/name;source=base/'InfiniOps';m=json.loads((base/'migration-manifest.json').read_text());assert m['ready']and m['pins']['InfiniOps']=='f998e4d6b445e29a950d048d0bc31f494ca7cce9';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for v in m['changes']:
 if 'merged_sha256'in v:assert sha(source/v['path'])==v['merged_sha256'],v['path']
if name=='Kimi':
 tile=json.loads((r/'results/tile-sdk-r2559-stage.json').read_text());assert tile['ready']and tile['pins']['InfiniOps']=='945184c02d767f9dfc343ba7a17f3409f111f32c'
 for path,h in tile['files'].items():assert sha(source/path)==h,path
sdk=r/'install'/f'hygon-{name.lower()}-06d0c03-r3550';build=r/'build'/f'ops-{name.lower()}-06d0c03-r3550';assert not sdk.exists()and not build.exists();parent=r/'install'/f'hygon-{name.lower()}-3e0d89fe-r2844';old=json.loads((r/'results'/f'latest-sdk-r2844-{name}-build.json').read_text());assert old['completed']and sha(parent/'lib/libinfiniops.so')==old['library_sha256'];assert sha(parent/'lib/libinfinirt.so')=='92b8eb6c419e17002bdcc47360b97b9bca87f19dcc62effabeea08d5f23df342';common=r/'install/d520-common-r1889';ccl=json.loads((r/'results/latest-ccl-r1889-build.json').read_text());assert ccl['completed']and sha(common/'lib/libinfiniccl.so')==ccl['library_sha256']
sdk.mkdir();(sdk/'include/infini').mkdir(parents=True);shutil.copytree(parent/'include/infini/rt',sdk/'include/infini/rt',symlinks=True);shutil.copy2(parent/'include/infini/rt.h',sdk/'include/infini/rt.h');shutil.copytree(common/'include/infiniccl',sdk/'include/infiniccl',symlinks=True);(sdk/'lib').mkdir();reused={}
for p in(parent/'lib').iterdir():
 if p.name.startswith(('libinfiniops','libinfiniccl'))or p.is_dir():continue
 dest=sdk/'lib'/p.name
 if p.is_symlink():dest.symlink_to(p.readlink())
 else:shutil.copy2(p,dest);reused[p.name]=sha(p)
for p in(common/'lib').glob('libinfiniccl*'):
 dest=sdk/'lib'/p.name
 if p.is_symlink():dest.symlink_to(p.readlink())
 else:shutil.copy2(p,dest)
generated=source/'generated';generated_backup=None
if generated.exists():
 backup=base/'PARENT_GENERATED_r3550';assert not backup.exists();generated.rename(backup);generated_backup=str(backup)
values={}
for line in(r/'build'/f'ops-{name.lower()}-3e0d89fe-r2844/CMakeCache.txt').read_text().splitlines():
 if not line or line.startswith(('#','//'))or'='not in line or':'not in line.split('=')[0]:continue
 kt,value=line.split('=',1);key,kind=kt.split(':',1)
 if key.startswith(('WITH_','INFINI_OPS_','FETCHCONTENT_SOURCE_DIR_'))or key in {'FETCHCONTENT_FULLY_DISCONNECTED','FETCHCONTENT_UPDATES_DISCONNECTED','CMAKE_BUILD_TYPE','CMAKE_CUDA_ARCHITECTURES','CMAKE_CUDA_COMPILER','CMAKE_CUDA_FLAGS','CMAKE_CXX_COMPILER','CMAKE_CXX_FLAGS','CMAKE_CUDA_SEPARABLE_COMPILATION','CUDAToolkit_ROOT','HYGON_ARCH','INFINI_RT_ROOT','INFINI_RT_INCLUDE_DIRS','INFINI_RT_LIBRARY','INFINI_RT_SOURCE_DIR','INFINIOPS_DTK_OPENMP','INFINIOPS_HIP_RUNTIME'}:
  if kind not in {'INTERNAL','STATIC'}:values[kt]=value
values['CMAKE_INSTALL_PREFIX:PATH']=str(sdk)
values['BUILD_NVIDIA_FLASH_INFERENCE:BOOL']='OFF';values['BUILD_NVIDIA_W4_MARLIN:BOOL']='OFF'
assert values.get('WITH_HYGON:BOOL')=='ON'and values.get('WITH_NVIDIA:BOOL','OFF')=='OFF'
record={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'variant':name,'source':str(source),'build':str(build),'install':str(sdk),'pins':m['pins'],'migration_manifest_sha256':sha(base/'migration-manifest.json'),'reused_unchanged_pinned_libraries':reused,'RT_sha256':sha(sdk/'lib/libinfinirt.so'),'CCL_sha256':sha(sdk/'lib/libinfiniccl.so'),'configuration':values,'generated_backup':generated_backup,'scope':'FreshbcfeOps with isolatednativeHygonPad3528, guard-preserving nativecontract compile with qualifiedF64Cast source retained with latestrecursive sourcepins/freshgeneratedwrappers/freshprefix. Onlyunchangedb413RTand6fCCLheaders/binaries copiedandSHAverified. No parentOpsheaders/binaries copied; wholeGPU/model correctness unproven.'};out=r/'results'/f'latest-sdk-r3550-{name}-configure.json';assert not out.exists();out.write_text(json.dumps(record,indent=2));subprocess.run(['cmake','-S',str(source),'-B',str(build)]+['-D'+k+'='+v for k,v in values.items()],check=True)
