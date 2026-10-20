import pathlib,json,sys,hashlib,shutil,subprocess,datetime
r=pathlib.Path('/refactor');name=sys.argv[1];assert name in ['Kimi','GLM','Deep'];root=r/'candidates/rebased-5f91ad7-r3668/Ops'/name;source=root/'InfiniOps';m=json.loads((root/'migration-manifest.json').read_text());assert m['ready'];runtime=json.loads((r/'results/tip-runtime-r3669-build.json').read_text());assert runtime['completed'];pins={k:v for k,v in runtime['pins'].items()if k!='ci'};assert m['pins']==pins;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for v in m['changes']:
 if'merged_sha256'in v:assert sha(source/v['path'])==v['merged_sha256'],v['path']
sdk=r/'install'/f'hygon-{name.lower()}-5f91ad7-r3672';build=r/'build'/f'ops-{name.lower()}-5f91ad7-r3672';common=r/'install/stack-5f91ad7-r3669-common';assert not sdk.exists()and not build.exists();sdk.mkdir();shutil.copytree(common/'include',sdk/'include',symlinks=True);(sdk/'lib').mkdir()
for pattern in ['libinfinirt*','libinfiniccl*']:
 for p in(common/'lib').glob(pattern):
  dest=sdk/'lib'/p.name
  if p.is_symlink():dest.symlink_to(p.readlink())
  else:shutil.copy2(p,dest)
assert sha(sdk/'lib/libinfinirt.so')==runtime['RT_sha256']and sha(sdk/'lib/libinfiniccl.so')==runtime['CCL_sha256'];assert not(source/'generated').exists()
values={}
for line in(r/'build'/f'ops-{name.lower()}-06d0c03-r3550/CMakeCache.txt').read_text().splitlines():
 if not line or line.startswith(('#','//'))or'='not in line or':'not in line.split('=')[0]:continue
 kt,value=line.split('=',1);key,kind=kt.split(':',1)
 if key.startswith(('WITH_','INFINI_OPS_','FETCHCONTENT_SOURCE_DIR_'))or key in {'FETCHCONTENT_FULLY_DISCONNECTED','FETCHCONTENT_UPDATES_DISCONNECTED','CMAKE_BUILD_TYPE','CMAKE_CUDA_ARCHITECTURES','CMAKE_CUDA_COMPILER','CMAKE_CUDA_FLAGS','CMAKE_CXX_COMPILER','CMAKE_CXX_FLAGS','CMAKE_CUDA_SEPARABLE_COMPILATION','CUDAToolkit_ROOT','HYGON_ARCH','INFINI_RT_ROOT','INFINI_RT_INCLUDE_DIRS','INFINI_RT_LIBRARY','INFINI_RT_SOURCE_DIR','INFINIOPS_DTK_OPENMP','INFINIOPS_HIP_RUNTIME'}:
  if kind not in{'INTERNAL','STATIC'}:values[kt]=value
values['CMAKE_INSTALL_PREFIX:PATH']=str(sdk)
for k,v in list(values.items()):
 key=k.split(':')[0]
 if key=='INFINI_RT_ROOT':values[k]=str(sdk)
 elif key=='INFINI_RT_INCLUDE_DIRS':values[k]=str(sdk/'include')
 elif key=='INFINI_RT_LIBRARY':values[k]=str(sdk/'lib/libinfinirt.so')
 elif key=='INFINI_RT_SOURCE_DIR'or key=='FETCHCONTENT_SOURCE_DIR_INFINIRT':values[k]=str(next(pathlib.Path(v['remote_path'])for v in json.loads((r/'results/upstream-5f91ad7-r3667.json').read_text())['repositories']if v['repo']=='InfiniTensor/InfiniRT')).replace('/path/to/cluster-home/hygon-inference-refactor-20261014','/refactor')
values['BUILD_NVIDIA_FLASH_INFERENCE:BOOL']='OFF';values['BUILD_NVIDIA_W4_MARLIN:BOOL']='OFF';assert values.get('WITH_HYGON:BOOL')=='ON'and values.get('WITH_NVIDIA:BOOL','OFF')=='OFF'
out=r/'results'/f'latest-sdk-r3672-{name}-configure.json';assert not out.exists();d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'variant':name,'pins':pins,'source':str(source),'build':str(build),'install':str(sdk),'configuration':values,'RT_sha256':runtime['RT_sha256'],'CCL_sha256':runtime['CCL_sha256'],'scope':'Fresh5f91/2b4/1ca Hygon SDK; newlyrebuilt cd1RT/51ccCCL exactSHA copied intoownprefix. No oldRT/CCL/Ops binaries. All math andmodels requirequalification.'};out.write_text(json.dumps(d,indent=2));subprocess.run(['cmake','-S',str(source),'-B',str(build)]+['-D'+k+'='+v for k,v in values.items()],check=True)
