import datetime,hashlib,json,pathlib,shutil,subprocess,sys
r=pathlib.Path('/refactor');name=sys.argv[1];s=json.loads((r/'results/cast64-sdks-r3315-stage.json').read_text());assert s['ready']and name in s['variants'];v=s['variants'][name];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def path(n):return r/pathlib.Path(n).relative_to(s['workspace_root'])
source,build,prefix,parent=[path(v[n])for n in['source','build','install','parent_install']];assert not build.exists()and not prefix.exists();assert sha(parent/'lib/libinfiniops.so')==v['parent_Op_sha256'];assert sha(parent/'lib/libinfinirt.so')==v['RT_sha256']and sha(parent/'lib/libinfiniccl.so')==v['CCL_sha256']
for n,h in v['source_sha256'].items():assert sha(source/n)==h,n
(prefix/'include/infini').mkdir(parents=True);shutil.copytree(parent/'include/infini/rt',prefix/'include/infini/rt',symlinks=True);shutil.copy2(parent/'include/infini/rt.h',prefix/'include/infini/rt.h');shutil.copytree(parent/'include/infiniccl',prefix/'include/infiniccl',symlinks=True);(prefix/'lib').mkdir()
for p in(parent/'lib').iterdir():
 if p.name.startswith('libinfiniops')or p.is_dir():continue
 if p.is_symlink():(prefix/'lib'/p.name).symlink_to(p.readlink())
 else:shutil.copy2(p,prefix/'lib'/p.name)
assert sha(prefix/'lib/libinfinirt.so')==v['RT_sha256']and sha(prefix/'lib/libinfiniccl.so')==v['CCL_sha256']
values=dict(v['configure']);values['CMAKE_INSTALL_PREFIX:PATH']=str(prefix);assert values.get('WITH_HYGON:BOOL')=='ON'and values.get('WITH_NVIDIA:BOOL','OFF')=='OFF'
out=r/'results'/f'cast64-sdk-r3316-{name}-configure.json';assert not out.exists();d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'variant':name,'pins':v['pins'],'source':str(source),'build':str(build),'install':str(prefix),'parent_install':str(parent),'configuration':values,'source_stage_sha256':sha(r/'results/cast64-sdks-r3315-stage.json'),'configured':False};out.write_text(json.dumps(d,indent=2));subprocess.run(['cmake','-S',str(source),'-B',str(build)]+['-D'+k+'='+value for k,value in values.items()],check=True);d['configured']=True;out.write_text(json.dumps(d,indent=2));print('CAST64_SDK_R3316_CONFIGURED',name,flush=True)
