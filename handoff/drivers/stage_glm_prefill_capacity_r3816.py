import pathlib,json,hashlib,datetime,shutil
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';src=r/'candidates/glm-shared-rope-5f91ad7-r3783';dst=r/'candidates/glm-prefill-capacity-5f91ad7-r3816';assert not dst.exists();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();proof=json.loads((r/'results/glm-shared-rope-r3783-build.json').read_text());component=json.loads((r/'results/glm-capcache-r3814-audit.json').read_text());assert proof['completed']and component['completed'];assert sha(src/'libmodern_glm_shared_rope_r3783.so')==proof['library_sha256'];shutil.copytree(src,dst,ignore=shutil.ignore_patterns('libmodern_glm_shared_rope_r3783.so'))
shutil.copy2(r/'candidates/glm-capcache-r3806/libglm_capcache_r3806.so',dst/'helpers/libglm_capcache_r3806.so');shutil.copy2(r/'incoming/glm_fp8_capcache_r3806.hpp',dst/'src/glm_fp8_capcache_r3806.hpp')
p=dst/'src/glm_fp8_runtime_r809.cpp';s=p.read_text();s='#include "glm_fp8_capcache_r3806.hpp"\n'+s
s=s.replace('bool batch_enabled{false};','void*capacity{};bool prefill_enabled{false};uint64_t prefill_switches{};bool batch_enabled{false};')
s=s.replace('~Owner(){if(handle', '~Owner(){if(capacity&&glm_fp8_capcache_destroy_r3806(capacity,runtime,stream))std::terminate();if(handle')
s=s.replace('std::array<std::array<uint64_t,10>,8>released{};','std::array<std::array<uint64_t,10>,8>released{};\n std::array<std::array<uint64_t,12>,8>released_capacity{};')
marker='namespace infinilm::models::glm_moe_dsa {\n';assert s.count(marker)==1
s=s.replace(marker,marker+'''
void glm_prefill_capacity_policy_r3816(rt::Runtime&r,int enabled){
 r.assertOwnerThread();if(r.isGraphRecording()||enabled<0||enabled>1)throw std::invalid_argument("3816 owner policy");auto&c=owner(r);
 if(!enabled&&c.capacity){r.syncStream();if(glm_fp8_capcache_clear_r3806(c.capacity,&r,r.stream()))throw std::runtime_error(glm_fp8_capcache_error_r3806());}
 c.prefill_enabled=enabled;++c.prefill_switches;
}
void glm_prefill_capacity_stats_r3816(rt::Runtime&r,uint64_t*out,size_t count){
 r.assertOwnerThread();if(!out||count!=14)throw std::invalid_argument("3816 stats geometry");std::fill(out,out+count,0);auto i=store.owners.find(&r);if(i==store.owners.end())return;auto&c=*i->second;
 if(c.device!=r.device().index()||c.stream!=r.stream())throw std::logic_error("3816 changed owner");out[0]=c.prefill_enabled;out[1]=c.prefill_switches;
 if(c.capacity&&glm_fp8_capcache_stats_r3806(c.capacity,out+2,12))throw std::runtime_error(glm_fp8_capcache_error_r3806());
}
bool fp8_capacity_r3816(rt::Runtime&r,const rt::Tensor&out,const rt::Tensor&x,const rt::Tensor&xs,const rt::Tensor&w,const rt::Tensor&ws,size_t ro,size_t co){
 r.assertOwnerThread();auto it=store.owners.find(&r);if(x->size(0)<=64||it==store.owners.end()||!it->second->prefill_enabled)return false;
 if(r.isGraphRecording()||r.device().type()!=rt::Device::Type::kHygon||out->dtype()!=rt::DataType::kBFloat16||x->dtype()!=rt::DataType::kUInt8||w->dtype()!=x->dtype()||xs->dtype()!=rt::DataType::kFloat32||ws->dtype()!=xs->dtype()||out->device()!=r.device()||x->device()!=r.device()||xs->device()!=r.device()||w->device()!=r.device()||ws->device()!=r.device()||!out->is_contiguous()||!x->is_contiguous()||!xs->is_contiguous()||!w->is_contiguous()||!ws->is_contiguous()||out->ndim()!=2||x->ndim()!=2||w->ndim()!=2||xs->ndim()!=2||ws->ndim()!=2||out->size(0)!=x->size(0)||out->size(1)!=w->size(0)||x->size(1)!=w->size(1)||xs->size(0)!=x->size(0)||xs->size(1)!=ws->size(1))throw std::invalid_argument("3816 owned contiguous FP8 geometry differs");
 size_t m=x->size(0),n=w->size(0),k=x->size(1),sr=ws->size(0),sc=ws->size(1);if(!glm_fp8_capcache_fits_r3806(m,n,k,sr,sc,ro,co))return false;
 auto&c=owner(r);if(!c.capacity){c.capacity=glm_fp8_capcache_create_r3806(c.device,c.stream,&r);if(!c.capacity)throw std::runtime_error(glm_fp8_capcache_error_r3806());}
 if(glm_fp8_capcache_run_r3806(c.capacity,&r,c.stream,m,n,k,sr,sc,ro,co,out->data(),x->data(),xs->data(),w->data(),ws->data()))throw std::runtime_error(glm_fp8_capcache_error_r3806());modern_glm_fp8_count_r780(c.device);return true;
}
''')
needle='r.assertOwnerThread();size_t m=x->size(0);';assert s.count(needle)==1;s=s.replace(needle,needle+'\n if(fp8_capacity_r3816(r,out,x,xs,w,ws,ro,co))return true;')
needle='store.owners.erase(i);return 0;';assert s.count(needle)==1;s=s.replace(needle,'''if(p.capacity){if(glm_fp8_capcache_clear_r3806(p.capacity,r,p.stream))throw std::runtime_error(glm_fp8_capcache_error_r3806());auto&v=store.released_capacity.at(p.device);if(glm_fp8_capcache_stats_r3806(p.capacity,v.data(),v.size()))throw std::runtime_error(glm_fp8_capcache_error_r3806());if(v[0]!=v[1]||v[5]||v[6])throw std::logic_error("3816 live capacity plans onclear");}
 store.owners.erase(i);return 0;''')
s+='''
extern "C" int modern_glm_prefill_capacity_released_r3816(int device,uint64_t*out,size_t count){try{if(device<0||device>=8||!out||count!=12||!store.owners.empty())throw std::invalid_argument("3816 released requires closed local owners");std::copy(store.released_capacity[device].begin(),store.released_capacity[device].end(),out);return 0;}catch(const std::exception&e){error=e.what();return 1;}}
''';p.write_text(s)
p=dst/'src/glm_fp8_runtime_r809.hpp';p.write_text(p.read_text()+'''
extern "C" int modern_glm_prefill_capacity_released_r3816(int,uint64_t*,size_t);
namespace infinilm::models::glm_moe_dsa {
void glm_prefill_capacity_policy_r3816(runtime::Runtime&,int);
void glm_prefill_capacity_stats_r3816(runtime::Runtime&,uint64_t*,size_t);
}
''')
p=dst/'src/modern_glm_cached_model_bridge_r809.cpp';p.write_text(p.read_text()+'\n'+(r/'incoming/glm_prefill_capacity_bridge_r3816.cpp').read_text())
p=dst/'compile.sh';s=p.read_text().replace('candidates/glm-shared-rope-5f91ad7-r3783','candidates/glm-prefill-capacity-5f91ad7-r3816').replace('libmodern_glm_shared_rope_r3783.so','libmodern_glm_prefill_capacity_r3816.so');s=s.replace('-lglm_fp8_cache_r807','-lglm_capcache_r3806 -lglm_fp8_cache_r807');p.write_text(s)
out=r/'results/glm-prefill-capacity-r3816-stage.json';assert not out.exists();out.write_text(json.dumps({'ready':True,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pins':proof['pins'],'parent_build_sha256':sha(r/'results/glm-shared-rope-r3783-build.json'),'component_audit_sha256':sha(r/'results/glm-capcache-r3814-audit.json'),'files_sha256':{p.relative_to(dst).as_posix():sha(p)for p in dst.rglob('*')if p.is_file()and p.suffix in['.cpp','.hpp','.h','.sh']},'helper_sha256':sha(dst/'helpers/libglm_capcache_r3806.so'),'scope':'DefaultOFF owner-capacity FP8 plan reuse only M>64, actualM math identical776; bounded768MiB16LRU, old192MiBcache retained. Fulloriginal1M RoPE sharing retained. Perowner idlepolicy +releasedstats; no weights/SDK/kernel/math change, no performance inheritance.'},indent=2));print(json.dumps({'staged':True}))
