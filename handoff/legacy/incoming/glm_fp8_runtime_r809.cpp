#include "glm_fp8_runtime_r809.hpp"
#include <algorithm>
#include <array>
#include <map>
#include <memory>
namespace rt=infinilm::runtime;
namespace {
thread_local std::string error;
struct Owner {
 rt::Runtime*runtime{};int device{};void*stream{};void*handle{};
 ~Owner(){if(handle&&glm_fp8_cache_destroy_r807(handle,runtime,stream))std::terminate();}
};
struct Store {
 std::map<rt::Runtime*,std::unique_ptr<Owner>>owners;
 std::array<std::array<uint64_t,10>,8>released{};
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
bool fp8_cached_r809(rt::Runtime&r,const rt::Tensor&out,const rt::Tensor&x,const rt::Tensor&xs,const rt::Tensor&w,const rt::Tensor&ws,size_t ro,size_t co){
 r.assertOwnerThread();size_t m=x->size(0);
 if(m>=32)return fp8_prefill_r780(r,out,x,xs,w,ws,ro,co);
 if(m!=8&&m!=16)return false;
 if(r.isGraphRecording()||r.device().type()!=rt::Device::Type::kHygon||out->dtype()!=rt::DataType::kBFloat16||x->dtype()!=rt::DataType::kUInt8||w->dtype()!=x->dtype()||xs->dtype()!=rt::DataType::kFloat32||ws->dtype()!=xs->dtype()||out->device()!=r.device()||x->device()!=r.device()||xs->device()!=r.device()||w->device()!=r.device()||ws->device()!=r.device()||!out->is_contiguous()||!x->is_contiguous()||!xs->is_contiguous()||!w->is_contiguous()||!ws->is_contiguous()||out->ndim()!=2||x->ndim()!=2||w->ndim()!=2||xs->ndim()!=2||ws->ndim()!=2||out->size(0)!=m||out->size(1)!=w->size(0)||x->size(1)!=w->size(1)||xs->size(0)!=m||xs->size(1)!=ws->size(1))throw std::invalid_argument("809 owned contiguous FP8 M8/16 geometry differs");
 auto&c=owner(r);
 if(glm_fp8_cache_run_r807(c.handle,&r,r.stream(),1,1,m,w->size(0),x->size(1),ws->size(0),ws->size(1),ro,co,const_cast<std::byte*>(out->data()),x->data(),xs->data(),w->data(),ws->data()))throw std::runtime_error(glm_fp8_cache_error_r807());
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
 store.owners.erase(i);return 0;
 }catch(const std::exception&e){error=e.what();return 1;}}
