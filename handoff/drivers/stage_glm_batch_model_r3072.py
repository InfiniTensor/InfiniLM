import datetime,hashlib,json,pathlib,shutil
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';parent=r/'candidates/models-3e0d89fe-r2856/GLM';target=r/'candidates/glm-batch-model-r3072';out=r/'results/glm-batch-model-r3072-stage.json';assert not target.exists()and not out.exists()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();proof=json.loads((r/'results/model-latest-r2856-GLM-build.json').read_text());assert proof['completed']and sha(parent/'libmodern_glm_latest_r2856.so')==proof['library_sha256']
# Same exact current pinned GPU objects/math helpers are reused and hashed;
# this experiment changes host plan lifetime only, not GPU arithmetic.
shutil.copytree(parent,target)
for name,h in proof['fresh_GPU_objects'].items():assert sha(target/name)==h
for name,h in proof['fresh_helpers'].items():assert sha(target/name)==h
before={p.relative_to(target).as_posix():sha(p)for p in target.rglob('*')if p.is_file()}
for src,dst in [('glm_fp8_runtime_batch_r3068.cpp','src/glm_fp8_runtime_r809.cpp'),('glm_batch_cache_geometry_r3068.hpp','src/glm_batch_cache_geometry_r3068.hpp'),('glm_fp8_cache_batch_r3039.cpp','helpers/src/glm_fp8_cache_r807.cpp')]:shutil.copy2(r/'incoming'/src,target/dst)
p=target/'src/modern_glm_cached_model_bridge_r809.cpp';p.write_text(p.read_text()+(r/'incoming/glm_batch_cache_bridge_r3068.inc').read_text())
cmd=(target/'compile.sh').read_text().replace('/refactor/candidates/models-3e0d89fe-r2856/GLM','/refactor/candidates/glm-batch-model-r3072').replace('libmodern_glm_latest_r2856.so','libmodern_glm_batch_r3072.so');(target/'compile.sh').write_text(cmd)
changed=[name for name,h in before.items()if sha(target/name)!=h];assert set(changed)=={'src/glm_fp8_runtime_r809.cpp','helpers/src/glm_fp8_cache_r807.cpp','src/modern_glm_cached_model_bridge_r809.cpp','compile.sh'},changed
new=set(p.relative_to(target).as_posix()for p in target.rglob('*')if p.is_file())-set(before);assert new=={'src/glm_batch_cache_geometry_r3068.hpp'},new
d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'ready':True,'parent_library_sha256':proof['library_sha256'],'before_sha256':before,'changed':changed,'added':sorted(new),'pins':proof['pins'],'source_sha256':{p.relative_to(target).as_posix():sha(p)for p in target.rglob('*')if p.is_file()and p.suffix in['.cpp','.hpp','.cu','.sh']},'scope':'Isolated copy of exact current2856. Original helperGPUobjects and math binaries reused/hashchecked; host runtime adapter adds defaultOFF owner/idle/reset policy and dispatchstats, boundedM32/64 planreuse. Originalmodel nevermutated. Not freshGPUcompile or fullmodel qualification.'};out.write_text(json.dumps(d,indent=2));print(json.dumps({'ready':True,'changed':changed,'added':sorted(new)}))
