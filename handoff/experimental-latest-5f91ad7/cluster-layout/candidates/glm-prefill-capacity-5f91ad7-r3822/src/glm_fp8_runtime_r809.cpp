#include "glm_fp8_capcache_r3806.hpp"
#include "glm_fp8_runtime_r809.hpp"
#include "glm_batch_cache_geometry_r3068.hpp"
#include <algorithm>
#include <array>
#include <map>
#include <memory>
namespace rt=infinilm::runtime;
namespace {
thread_local std::string error;
struct Owner {
 rt::Runtime*runtime{};int device{};void*stream{};void*handle{};void*capacity{};bool prefill_enabled{false};uint64_t prefill_switches{};bool batch_enabled{false};uint64_t batch32{},batch64{},batch_fallback{},batch_switches{};
 ~Owner(){if(capacity&&glm_fp8_capcache_destroy_r3806(capacity,runtime,stream))std::terminate();if(handle&&glm_fp8_cache_destroy_r807(handle,runtime,stream))std::terminate();}
};
struct Store {
 std::map<rt::Runtime*,std::unique_ptr<Owner>>owners;
 std::array<std::array<uint64_t,10>,8>released{};
 std::array<std::array<uint64_t,12>,8>released_capacity{};
 ~Store(){if(!owners.empty())std::terminate();}
};
thread_local Store store;
Owner&owner(rt::Runtime&r){
 auto i=store.owners.find(&r);
 if(i==store.owners.end()){
  auto p=std::make_unique<Owner>();p->runtime=&r;p->device=r.device().index();p->stream=r.stream();p->handle=glm_fp8_cache_create_r807(p->device,p->stream,&r);
  if(!p->handle)throw std::runtime_error(glm_fp8_cache_error_r807());
  i=store.owners.emplace(&r,std::move(p)).first;
 }
 if(i->second->stream!=r.stream()||i->second->device!=r.device().index())throw std::invalid_argument("809 Runtime stream/device changed");
 return*i->second;
}
}
namespace infinilm::models::glm_moe_dsa {

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
 if(glm_fp8_capcache_run_r3806(c.capacity,&r,c.stream,m,n,k,sr,sc,ro,co,const_cast<std::byte*>(out->data()),x->data(),xs->data(),w->data(),ws->data()))throw std::runtime_error(glm_fp8_capcache_error_r3806());modern_glm_fp8_count_r780(c.device);return true;
}
void glm_fp8_batch_policy_r3068(rt::Runtime&r,bool enabled){
 r.assertOwnerThread();if(r.isGraphRecording())throw std::logic_error("3068 policy during graph");
 auto&c=owner(r);c.batch_enabled=enabled;++c.batch_switches;
}
void glm_fp8_batch_stats_r3068(rt::Runtime&r,uint64_t*out,size_t count){
 r.assertOwnerThread();if(!out||count!=5)throw std::invalid_argument("3068 stats geometry");
 auto i=store.owners.find(&r);if(i==store.owners.end()){std::fill(out,out+5,0);return;}
 auto&c=*i->second;if(c.stream!=r.stream()||c.device!=r.device().index())throw std::logic_error("3068 stats changed owner");
 const std::array<uint64_t,5>v{uint64_t(c.batch_enabled),c.batch32,c.batch64,c.batch_fallback,c.batch_switches};std::copy(v.begin(),v.end(),out);
}
bool fp8_cached_r809(rt::Runtime&r,const rt::Tensor&out,const rt::Tensor&x,const rt::Tensor&xs,const rt::Tensor&w,const rt::Tensor&ws,size_t ro,size_t co){
 r.assertOwnerThread();size_t m=x->size(0);
 if(fp8_capacity_r3816(r,out,x,xs,w,ws,ro,co))return true;
 if(m>=32){
  auto it=store.owners.find(&r);const bool enabled=it!=store.owners.end()&&it->second->batch_enabled;
  const bool admitted=enabled&&x->ndim()==2&&w->ndim()==2&&ws->ndim()==2&&
    glm_batch_cache_r3068::fits(m,w->size(0),x->size(1),ws->size(0),ws->size(1),ro,co);
  if(!admitted){if(enabled)++it->second->batch_fallback;return fp8_prefill_r780(r,out,x,xs,w,ws,ro,co);}
 }
 if(m!=8&&m!=16&&m!=32&&m!=64)return false;
 if(r.isGraphRecording()||r.device().type()!=rt::Device::Type::kHygon||out->dtype()!=rt::DataType::kBFloat16||x->dtype()!=rt::DataType::kUInt8||w->dtype()!=x->dtype()||xs->dtype()!=rt::DataType::kFloat32||ws->dtype()!=xs->dtype()||out->device()!=r.device()||x->device()!=r.device()||xs->device()!=r.device()||w->device()!=r.device()||ws->device()!=r.device()||!out->is_contiguous()||!x->is_contiguous()||!xs->is_contiguous()||!w->is_contiguous()||!ws->is_contiguous()||out->ndim()!=2||x->ndim()!=2||w->ndim()!=2||xs->ndim()!=2||ws->ndim()!=2||out->size(0)!=m||out->size(1)!=w->size(0)||x->size(1)!=w->size(1)||xs->size(0)!=m||xs->size(1)!=ws->size(1))throw std::invalid_argument("809 owned contiguous FP8 M8/16 geometry differs");
 auto&c=owner(r);
 if(glm_fp8_cache_run_r807(c.handle,&r,r.stream(),1,1,m,w->size(0),x->size(1),ws->size(0),ws->size(1),ro,co,const_cast<std::byte*>(out->data()),x->data(),xs->data(),w->data(),ws->data()))throw std::runtime_error(glm_fp8_cache_error_r807());
 if(m==32)++c.batch32;else if(m==64)++c.batch64;
 modern_glm_fp8_count_r780(r.device().index());return true;
}
}
extern "C" const char*modern_glm_fp8_runtime_error_r809(){return error.c_str();}
extern "C" int modern_glm_fp8_runtime_stats_r809(int device,uint64_t*out,size_t count){try{
 if(device<0||device>=8||!out||count!=10)throw std::invalid_argument("809 stats bounds");
 const Owner*found=nullptr;
 for(const auto&[identity,p]:store.owners)if(p->device==device){identity->assertOwnerThread();if(found)throw std::logic_error("809 ambiguous Runtime owner");found=p.get();}
 if(found){if(glm_fp8_cache_stats_r807(found->handle,out,count))throw std::runtime_error(glm_fp8_cache_error_r807());}
 else std::copy(store.released[device].begin(),store.released[device].end(),out);
 return 0;
 }catch(const std::exception&e){error=e.what();return 1;}}
extern "C" int modern_glm_fp8_runtime_clear_r809(rt::Runtime*r){try{
 if(!r)throw std::invalid_argument("809 missing Runtime");r->assertOwnerThread();auto i=store.owners.find(r);if(i==store.owners.end())return 0;
 auto&p=*i->second;r->syncStream();if(glm_fp8_cache_clear_r807(p.handle,r,p.stream))throw std::runtime_error(glm_fp8_cache_error_r807());
 auto&counts=store.released.at(p.device);if(glm_fp8_cache_stats_r807(p.handle,counts.data(),counts.size()))throw std::runtime_error(glm_fp8_cache_error_r807());
 if(counts[0]!=counts[1]||counts[5]||counts[6])throw std::logic_error("809 plans not fully released");
 if(p.capacity){if(glm_fp8_capcache_clear_r3806(p.capacity,r,p.stream))throw std::runtime_error(glm_fp8_capcache_error_r3806());auto&v=store.released_capacity.at(p.device);if(glm_fp8_capcache_stats_r3806(p.capacity,v.data(),v.size()))throw std::runtime_error(glm_fp8_capcache_error_r3806());if(v[0]!=v[1]||v[5]||v[6])throw std::logic_error("3816 live capacity plans onclear");}
 store.owners.erase(i);return 0;
 }catch(const std::exception&e){error=e.what();return 1;}}

extern "C" int modern_glm_prefill_capacity_released_r3816(int device,uint64_t*out,size_t count){try{if(device<0||device>=8||!out||count!=12||!store.owners.empty())throw std::invalid_argument("3816 released requires closed local owners");std::copy(store.released_capacity[device].begin(),store.released_capacity[device].end(),out);return 0;}catch(const std::exception&e){error=e.what();return 1;}}
