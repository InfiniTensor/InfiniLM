#include "runtime/runtime.hpp"
#include "runtime/tensor.hpp"
#include "execution/submit.hpp"
#include <infini/rt/device_guard.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <thread>
#include <string>
namespace rt=infinilm::runtime;namespace ex=infinilm::execution;
namespace {thread_local std::string error;struct Context{std::shared_ptr<rt::Runtime>owner;std::thread::id thread;};
void pointer_device(const void*p,int device){cudaPointerAttributes a{};if(!p||cudaPointerGetAttributes(&a,p)!=cudaSuccess||a.device!=device)throw std::invalid_argument("4391 pointer differs from owner device");}
}
extern "C" const char*modern_expert_error_r4391(){return error.c_str();}
extern "C" void*modern_expert_create_r4391(int device){try{infini::rt::DeviceGuard guard{rt::Device::Type::kHygon,device,true};auto c=std::make_unique<Context>();c->owner=rt::Runtime::create({rt::Device::Type::kHygon,device});c->owner->enableOwnerDrainFailStop();c->thread=std::this_thread::get_id();return c.release();}catch(const std::exception&e){error=e.what();return nullptr;}}
extern "C" int modern_expert_forward_r4391(void*raw,size_t tokens,size_t experts,void*out,void*input,void*ids,void*weights,void*w13,void*s13,void*w2,void*s2,void*caller_stream){
 auto*c=static_cast<Context*>(raw);try{
  if(!c||c->thread!=std::this_thread::get_id()||!tokens||tokens>16||experts<16||experts>896)throw std::invalid_argument("4391 native component geometry/owner");auto&r=*c->owner;r.assertOwnerThread();infini::rt::DeviceGuard guard{r.device().type(),r.device().index(),true};
  for(auto*p:{out,input,ids,weights,w13,s13,w2,s2})pointer_device(p,r.device().index());if(cudaStreamSynchronize(static_cast<cudaStream_t>(caller_stream))!=cudaSuccess)throw std::runtime_error("4391 input producer did not drain");
  auto view=[&](void*p,rt::Shape shape,rt::DataType dtype){return rt::Tensor::from_blob(p,shape,dtype,r.device());};
  auto y=view(out,{tokens,3584},rt::DataType::kBFloat16),x=view(input,{tokens,3584},rt::DataType::kBFloat16),i=view(ids,{tokens,16},rt::DataType::kInt32),w=view(weights,{tokens,16},rt::DataType::kFloat32),a=view(w13,{experts,768,1792},rt::DataType::kUInt8),b=view(s13,{experts,768,112},rt::DataType::kUInt8),z=view(w2,{experts,3584,192},rt::DataType::kUInt8),s=view(s2,{experts,3584,12},rt::DataType::kUInt8);
  auto config=infini::ops::FusedMoeMxfp4::DefaultConfig(r.device().type());config.set_implementation_index(1);ex::submit<infini::ops::FusedMoeMxfp4>(r,ex::RecordingPolicy::AllowHostReplay,config,y,x,i,w,a,b,z,s,size_t{2});r.syncStream();return 0;
 }catch(const std::exception&e){error=e.what();if(c&&c->owner){try{c->owner->syncStream();}catch(...){std::terminate();}}return 1;}
}
extern "C" int modern_expert_destroy_r4391(void*raw){auto*c=static_cast<Context*>(raw);try{if(!c||c->thread!=std::this_thread::get_id())throw std::invalid_argument("4391 foreign destroy");auto dev=c->owner->device();infini::rt::DeviceGuard guard{dev.type(),dev.index(),true};c->owner->syncStream();c->owner->trimMemory();c->owner.reset();delete c;return 0;}catch(const std::exception&e){error=e.what();return 1;}}
