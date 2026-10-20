#include "runtime/runtime.hpp"
#include "execution/submit.hpp"
#include <infini/rt/device_guard.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>
namespace rt=infinilm::runtime;namespace ex=infinilm::execution;
namespace {thread_local std::string error;
struct Context{std::shared_ptr<rt::Runtime>r;};
rt::DataType dtype(int t){switch(t){case 0:return rt::DataType::kFloat64;case 1:return rt::DataType::kFloat32;case 2:return rt::DataType::kFloat16;case 3:return rt::DataType::kBFloat16;default:throw std::invalid_argument("3376 dtype");}}
}
extern "C" const char*cast64_contract_error_r3376(){return error.c_str();}
extern "C" void*cast64_contract_create_r3376(int device){try{if(device<0||device>=8)throw std::invalid_argument("3376 device");infini::rt::DeviceGuard guard{rt::Device::Type::kHygon,device,true};auto c=std::make_unique<Context>();c->r=rt::Runtime::create({rt::Device::Type::kHygon,device});c->r->enableOwnerDrainFailStop();return c.release();}catch(const std::exception&e){error=e.what();return nullptr;}}
extern "C" int cast64_contract_run_r3376(void*raw,int in_type,int out_type,size_t count,int geometry,const void*x,void*y,void*producer_stream){
 auto*c=static_cast<Context*>(raw);rt::Tensor input,output;bool owned=false;
 try{if(!c||count>8192||geometry<0||geometry>4||!x||!y||!producer_stream)throw std::invalid_argument("3376 context/count/geometry/pointers");auto&r=*c->r;r.assertOwnerThread();owned=true;if(r.isGraphRecording())throw std::logic_error("3376 eager contract");auto it=dtype(in_type),ot=dtype(out_type);
  infini::rt::DeviceGuard guard{r.device().type(),r.device().index(),true};if(cudaStreamSynchronize(static_cast<cudaStream_t>(producer_stream))!=cudaSuccess)throw std::runtime_error("3376 source sync");
  if((geometry==2||geometry==3)&&(!count||count%2))throw std::invalid_argument("3376 stride geometry");
  if(geometry==2){input=rt::Tensor::from_blob(const_cast<void*>(x),{2,count/2},it,r.device())->permute({1,0});output=rt::Tensor::from_blob(y,{count/2,2},ot,r.device());}
  else if(geometry==3){input=rt::Tensor::from_blob(const_cast<void*>(x),{count/2,2},it,r.device());output=rt::Tensor::from_blob(y,{2,count/2},ot,r.device())->permute({1,0});}
  else{input=rt::Tensor::from_blob(const_cast<void*>(x),{count},it,r.device());output=rt::Tensor::from_blob(y,{geometry==1?count+1:count},ot,geometry==4?rt::Device{rt::Device::Type::kHygon,(r.device().index()+1)%8}:r.device());}
  ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),input,output);r.syncStream();return 0;
 }catch(const std::exception&e){error=e.what();if(c&&owned)try{c->r->syncStream();}catch(...){std::terminate();}return 1;}
}
extern "C" int cast64_contract_destroy_r3376(void*raw,int*closed){auto*c=static_cast<Context*>(raw);try{if(!c||!closed)throw std::invalid_argument("3376 destroy");c->r->assertOwnerThread();c->r->syncStream();c->r->trimMemory();std::weak_ptr<rt::Runtime>weak=c->r;delete c;if(!weak.expired())throw std::logic_error("3376 Runtime retained");*closed=1;return 0;}catch(const std::exception&e){error=e.what();return 1;}}
