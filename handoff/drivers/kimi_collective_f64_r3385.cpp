#include "runtime/runtime.hpp"
#include "execution/submit.hpp"
#include <infini/rt/device_guard.h>
#include "engine/distributed/communication_group.hpp"
#include <cuda_runtime_api.h>
#include <cstring>
#include <map>
#include <string>
namespace rt=infinilm::runtime;
namespace {thread_local std::string error;
std::shared_ptr<rt::Runtime> worker(int d){static thread_local std::map<int,std::shared_ptr<rt::Runtime>> workers;auto &r=workers[d];if(!r){r=rt::Runtime::create({rt::Device::Type::kHygon,d});r->enableOwnerDrainFailStop();}return r;}
struct Context{std::shared_ptr<rt::Runtime> runtime;infinicclComm_t comm=nullptr;};
}
extern "C" const char *kimi_projection_r3385_error(){return error.c_str();}
extern "C" size_t kimi_projection_r3385_id_size(){return sizeof(infinicclUniqueId);}
extern "C" int kimi_projection_r3385_id(void *output){try{auto r=worker(0);infini::rt::DeviceGuard guard{r->device().type(),0,true};infinicclUniqueId id{};if(infinicclGetUniqueId(&id)!=infinicclSuccess)throw std::runtime_error("new CCL unique ID creation failed");std::memcpy(output,&id,sizeof(id));return 0;}catch(const std::exception &e){error=e.what();return 1;}}
extern "C" void *kimi_projection_r3385_create(int device,int size,int rank,const void *id){
 std::unique_ptr<Context> c;try{int current=-1;cudaGetDevice(&current);if(current!=device)throw std::invalid_argument("CCL owning GPU differs");
 c=std::make_unique<Context>();c->runtime=worker(device);infini::rt::DeviceGuard guard{c->runtime->device().type(),device,true};infinicclUniqueId value{};std::memcpy(&value,id,sizeof(value));
 if(infinicclCommInitRank(&c->comm,size,value,rank)!=infinicclSuccess)throw std::runtime_error("new CCL rank initialization failed");return c.release();
 }catch(const std::exception &e){error=e.what();return nullptr;}
}
extern "C" int kimi_projection_r3385_allreduce(void *raw,int dtype,size_t count,const void *input,void *output,void *stream){
 auto c=static_cast<Context*>(raw);rt::Tensor in,out;bool owned=false;
 try{if(!c||!count||count>16*7168||!input||!output||!stream)throw std::invalid_argument("3385 CCL metadata");c->runtime->assertOwnerThread();owned=true;if(cudaStreamSynchronize(static_cast<cudaStream_t>(stream))!=cudaSuccess)throw std::runtime_error("CCL input drain failed");auto &r=*c->runtime;
 if(dtype<0||dtype>3)throw std::invalid_argument("3385 dtype");auto dt=dtype==3?rt::DataType::kFloat64:dtype==0?rt::DataType::kFloat32:dtype==1?rt::DataType::kFloat16:rt::DataType::kBFloat16;
 in=rt::Tensor::from_blob(const_cast<void*>(input),{count},dt,r.device());out=rt::Tensor::from_blob(output,{count},dt,r.device());
 infinilm::engine::distributed::allreduce_(r,out,in,infinicclSum,c->comm);r.syncStream();return 0;
 }catch(const std::exception &e){error=e.what();if(c&&owned)try{c->runtime->syncStream();}catch(...){std::terminate();}return 1;}
}
extern "C" int kimi_projection_r3385_destroy(void *raw){auto c=static_cast<Context*>(raw);try{if(!c)throw std::invalid_argument("3385 context");c->runtime->assertOwnerThread();c->runtime->syncStream();infini::rt::DeviceGuard guard{c->runtime->device().type(),c->runtime->device().index(),true};if(infinicclCommDestroy(c->comm)!=infinicclSuccess)throw std::runtime_error("CCL destruction failed");delete c;return 0;}catch(const std::exception &e){error=e.what();return 1;}}

extern "C" int kimi_projection_r3385_wide(void*raw,int dtype,size_t count,const void*x,void*y,void*stream){
 auto*c=static_cast<Context*>(raw);rt::Tensor in,out,wide;bool owned=false;try{
  if(!c||dtype!=0||!count||count>16*7168||!x||!y||!stream)throw std::invalid_argument("3385 bounded FP32-to-FP64 owner/pointers");auto&r=*c->runtime;r.assertOwnerThread();owned=true;if(r.isGraphRecording())throw std::logic_error("3385 eager only");
  if(cudaStreamSynchronize(static_cast<cudaStream_t>(stream))!=cudaSuccess)throw std::runtime_error("3385 source drain");
  in=rt::Tensor::from_blob(const_cast<void*>(x),{count},rt::DataType::kFloat32,r.device());out=rt::Tensor::from_blob(y,{count},rt::DataType::kFloat32,r.device());wide=rt::Tensor::empty(r,{count},rt::DataType::kFloat64);
  infinilm::execution::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),in,wide);
  infinilm::engine::distributed::allreduce_(r,wide,wide,infinicclSum,c->comm);
  infinilm::execution::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),wide,out);r.syncStream();return 0;
 }catch(const std::exception&e){error=e.what();if(c&&owned)try{c->runtime->syncStream();}catch(...){std::terminate();}return 1;}
}
