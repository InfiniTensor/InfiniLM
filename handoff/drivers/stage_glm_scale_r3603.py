import pathlib,json,shutil,hashlib,datetime
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';src=r/'candidates/models-06d0c03-r3586/GLM';dst=r/'candidates/glm-scale-06d0c03-r3603';assert not dst.exists();shutil.copytree(src,dst);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();proof=json.loads((r/'results/model-latest-r3586-GLM-build.json').read_text());assert proof['completed'] and sha(src/'libmodern_glm_latest_r3586.so')==proof['library_sha256'];changes={}
def edit(rel,changeset):
 p=dst/rel;s=p.read_text();before=sha(p)
 for a,b in changeset:
  assert s.count(a)==1,(rel,a,s.count(a));s=s.replace(a,b)
 p.write_text(s);changes[rel]={'before':before,'after':sha(p)}
edit('src/modern_glm_cached_model_bridge_r809.cpp',[
 ('#include <cstring>','#include <cstring>\n#include <cstdlib>'),
 ('int stage=0;bool ready=false;','int stage=0,pp=4;bool ready=false;'),
 ('if(device<0||device>=8||stage<0||stage>=4)throw std::invalid_argument("GLM full-model requires TP8 PP4");','const char*setting=std::getenv("GLM_PP_SIZE_R3603");int pp=4;if(setting){if(std::string(setting)=="2")pp=2;else if(std::string(setting)!="4")throw std::invalid_argument("GLM PP size must be exactly2 or4");}if(device<0||device>=8||stage<0||stage>=pp)throw std::invalid_argument("GLM full-model requires TP8 PP2/4");'),
 ('c->stage=stage;c->capacity_r1406=capacity;','c->stage=stage;c->pp=pp;c->capacity_r1406=capacity;'),
 ('&c->world,32,world,stage*8+device','&c->world,8*pp,world,stage*8+device'),
 ('&c->control,32,control,stage*8+device','&c->control,8*pp,control,stage*8+device'),
 ('rank.pp_size=4;rank.pp_stage=stage;rank.world_size=32;','rank.pp_size=pp;rank.pp_stage=stage;rank.world_size=8*pp;'),
 ('if(c->stage==3){','if(c->stage+1==c->pp){')])
edit('src/glm_moe_dsa_balanced_text_model_r126.cpp',[
 ('(rank.pp_size!=1&&rank.pp_size!=4)','(rank.pp_size!=1&&rank.pp_size!=2&&rank.pp_size!=4)'),
 ('explicit TP1/8 PP1/4 topology','explicit TP1/8 PP1/2/4 topology'),
 ('const size_t boundaries[]={0,18,38,58,78};begin_=rank.pp_size==1?0:boundaries[rank.pp_stage];end_=rank.pp_size==1?78:boundaries[rank.pp_stage+1];','const size_t pp4[]={0,18,38,58,78},pp2[]={0,38,78};const size_t*boundaries=rank.pp_size==2?pp2:pp4;begin_=rank.pp_size==1?0:boundaries[rank.pp_stage];end_=rank.pp_size==1?78:boundaries[rank.pp_stage+1];')])
cmd=(dst/'compile.sh').read_text().replace('/refactor/candidates/models-06d0c03-r3586/GLM','/refactor/candidates/glm-scale-06d0c03-r3603').replace('libmodern_glm_latest_r3586.so','libmodern_glm_scale_r3603.so');(dst/'compile.sh').write_text(cmd)
d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'ready':True,'pins':proof['pins'],'changes':changes,'sources':{p.relative_to(dst).as_posix():sha(p)for p in dst.rglob('*')if p.is_file()and p.suffix in['.cpp','.hpp','.h','.sh']},'reused_latest_GPU_objects':{p.relative_to(dst).as_posix():sha(p)for p in dst.rglob('*.o')},'reused_latest_helpers':{p.relative_to(dst).as_posix():sha(p)for p in(dst/'helpers').glob('*.so')},'scope':'Host CAPI and full78stage boundaries only: TP8PP2/4, defaultPP4. PP2 layer38 must be fullindexer. Latest3586 GPUobjects/helpers untouched. No pureTP16 or EP support claimed. Build and fullmodel qualification required.'};out=r/'results/glm-scale-r3603-stage.json';assert not out.exists();out.write_text(json.dumps(d,indent=2));print(json.dumps({'ready':True,'changes':changes}))
