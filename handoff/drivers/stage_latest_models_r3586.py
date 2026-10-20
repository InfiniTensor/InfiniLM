import pathlib,json,hashlib,datetime,shutil,re
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';base=r/'candidates/models-06d0c03-r3586';out=r/'results/models-latest-r3586-stage.json';assert not base.exists()and not out.exists();base.mkdir();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();stage=json.loads((r/'results/rebased-06d0c03-r3548-stage.json').read_text());assert stage['ready']and not stage['conflicts'];d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'ready':False,'pins':stage['pins'],'variants':{},'scope':'Latest06d/046/f998 full-model assembly: Kimi3261 privategroup ABI, GLM3072 batchcache, Deep3443 selectable precision retained in sources. All GPU objects/helpers newly compiled against latest SDK/Common; no old binaries. F64 optional control, native default; exact final token acceptance. Build is not model qualification.'};allowed={'.cpp','.cc','.c','.h','.hh','.hpp','.cuh','.cu','.inc','.sh','.py'}
parents={'Kimi':('candidates/kimi-reduce-model-3e0d89fe-r3261','kimi-reduce-model-r3262-build.json','libmodern_kimi_reduce_r3261.so'),'GLM':('candidates/glm-batch-model-r3072','glm-batch-model-r3073-build.json','libmodern_glm_batch_r3072.so'),'Deep':('candidates/deep-wide-3e0d89fe-r3443','deep-wide-r3447-build.json','libmodern_deep_wide_r3443.so')}
def copy_sources(src,dst,skip=()):
 assert src.is_dir()
 for p in src.rglob('*'):
  relative=p.relative_to(src)
  if any(x in relative.parts for x in skip)or not p.is_file()or p.suffix not in allowed:continue
  q=dst/relative;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
for name,(parent_path,receipt,library)in parents.items():
 parent=r/parent_path;proof=json.loads((r/'results'/receipt).read_text());assert proof['completed']and sha(parent/library)==proof['library_sha256'];target=base/name;target.mkdir();copy_sources(parent,target,skip=('model-src','__pycache__'))
 if name=='Deep':copy_sources(r/'candidates/models-3e0d89fe-r2856/Deep/helpers',target/'helpers')
 assert (target/'helpers/src').is_dir()and not list(target.rglob('*.o'))and not list(target.rglob('*.so'))
 model=r/'candidates/rebased-06d0c03-r3548/Model'/name/'InfiniLM';common=r/'candidates/rebased-06d0c03-r3548/Common'/name/'InfiniLM';manifest=json.loads((model.parent/'migration-manifest.json').read_text());assert manifest['ready']and manifest['pins']==d['pins']
 ABI_headers={}
 for rel in['csrc/runtime/runtime.hpp','csrc/runtime/tensor.hpp','csrc/runtime/device.hpp','csrc/runtime/dtype.hpp']:
  if(model/rel).exists()and(common/rel).exists():assert sha(model/rel)==sha(common/rel),('Runtime_header_ABI_mismatch',name,rel);ABI_headers[rel]=sha(model/rel)
 sdk=json.loads((r/'results'/f'latest-sdk-r3550-{name}-build.json').read_text());rt=json.loads((r/'results'/f'common-latest-r3551-{name}-build.json').read_text());assert sdk['completed']and rt['completed']and sdk['pins']==rt['pins']==d['pins']
 cmd=(target/'compile.sh').read_text();cmd=cmd.replace('/refactor/'+parent_path,'/refactor/candidates/models-06d0c03-r3586/'+name).replace(library,f'libmodern_{name.lower()}_latest_r3586.so');(target/'compile.sh').write_text(cmd)
 legacy=[]
 for p in target.rglob('*'):
  if p.is_file()and p.suffix in allowed:
   for no,line in enumerate(p.read_text().splitlines(),1):
    if any(v in line for v in['PadConstant','MulScalar']):legacy.append({'file':p.relative_to(target).as_posix(),'line':no,'text':line[:300]})
 d['variants'][name]={'ready':True,'parent_build_sha256':sha(r/'results'/receipt),'parent_source':parent_path,'library':f'libmodern_{name.lower()}_latest_r3586.so','compile_sha256':sha(target/'compile.sh'),'inputs_sha256':{p.relative_to(target).as_posix():sha(p)for p in target.rglob('*')if p.is_file()},'private_model_csrc':{p.relative_to(model).as_posix():sha(p)for p in(model/'csrc').rglob('*')if p.is_file()},'private_model_manifest_sha256':sha(model.parent/'migration-manifest.json'),'Runtime_ABI_headers_exact':ABI_headers,'legacy_API_references':legacy,'CAPI_symbols':(['modern_kimi_engine_'+s+'_r2475'for s in['create','forward','partition','destroy','layout']]if name=='Kimi'else['modern_glm_model_create','modern_glm_model_forward','modern_glm_model_batch_cache_policy_r3068']if name=='GLM'else['modern_native_engine_create','modern_native_engine_forward','modern_deep_precision_policy_r3443'])};out.write_text(json.dumps(d,indent=2))
d['ready']=True;out.write_text(json.dumps(d,indent=2));print(json.dumps({'ready':True,'variants':{n:{'source_files':len(v['inputs_sha256']),'legacy_API_references':v['legacy_API_references']}for n,v in d['variants'].items()}}))
