#include "runtime/runtime.hpp"
#include "execution/submit.hpp"
#include "layers/linear/dense.hpp"
#include <infini/rt/device_guard.h>
#include "engine/distributed/communication_group.hpp"
#include <cuda_runtime_api.h>
#include <cstring>
#include <map>
#include <string>
namespace rt=infinilm::runtime;
namespace {thread_local std::string error;
std::shared_ptr<rt::Runtime> worker(int d){auto r=rt::Runtime::create({rt::Device::Type::kHygon,d});r->enableOwnerDrainFailStop();return r;}
struct Context{std::shared_ptr<rt::Runtime> runtime;infinicclComm_t comm=nullptr;};
}
extern "C" const char *kimi_projection_r3883_error(){return error.c_str();}
extern "C" size_t kimi_projection_r3883_id_size(){return sizeof(infinicclUniqueId);}
extern "C" int kimi_projection_r3883_id(void *output){try{auto r=worker(0);infini::rt::DeviceGuard guard{r->device().type(),0,true};infinicclUniqueId id{};if(infinicclGetUniqueId(&id)!=infinicclSuccess)throw std::runtime_error("new CCL unique ID creation failed");std::memcpy(output,&id,sizeof(id));return 0;}catch(const std::exception &e){error=e.what();return 1;}}
extern "C" void *kimi_projection_r3883_create(int device,int size,int rank,const void *id){
 std::unique_ptr<Context> c;try{int current=-1;cudaGetDevice(&current);if(current!=device)throw std::invalid_argument("CCL owning GPU differs");
 c=std::make_unique<Context>();c->runtime=worker(device);infini::rt::DeviceGuard guard{c->runtime->device().type(),device,true};infinicclUniqueId value{};std::memcpy(&value,id,sizeof(value));
 if(infinicclCommInitRank(&c->comm,size,value,rank)!=infinicclSuccess)throw std::runtime_error("new CCL rank initialization failed");return c.release();
 }catch(const std::exception &e){error=e.what();return nullptr;}
}
extern "C" int kimi_projection_r3883_allreduce(void *raw,int dtype,size_t count,const void *input,void *output,void *stream){
 auto c=static_cast<Context*>(raw);rt::Tensor in,out;
 try{if(!c)throw std::invalid_argument("null context");c->runtime->assertOwnerThread();if(cudaStreamSynchronize(static_cast<cudaStream_t>(stream))!=cudaSuccess)throw std::runtime_error("CCL input drain failed");auto &r=*c->runtime;infini::rt::DeviceGuard guard{r.device().type(),r.device().index(),true};
 auto dt=dtype==0?rt::DataType::kFloat32:dtype==1?rt::DataType::kFloat16:rt::DataType::kBFloat16;
 in=rt::Tensor::from_blob(const_cast<void*>(input),{count},dt,r.device());out=rt::Tensor::from_blob(output,{count},dt,r.device());
 infinilm::engine::distributed::allreduce_(r,out,in,infinicclSum,c->comm);r.syncStream();return 0;
 }catch(const std::exception &e){error=e.what();try{c->runtime->syncStream();}catch(...){std::terminate();}return 1;}
}
extern "C" int kimi_projection_r3883_destroy(void *raw){auto c=static_cast<Context*>(raw);try{c->runtime->syncStream();infini::rt::DeviceGuard guard{c->runtime->device().type(),c->runtime->device().index(),true};if(infinicclCommDestroy(c->comm)!=infinicclSuccess)throw std::runtime_error("CCL destruction failed");delete c;return 0;}catch(const std::exception &e){error=e.what();return 1;}}

extern "C" int kimi_projection_r3883_gemm(void*raw,size_t m,int packed,const void*x,const void*w,void*y,void*input_stream){
 auto*c=static_cast<Context*>(raw);rt::Tensor a,b,out;try{
  if(!c||!x||!w||!y||!input_stream||(m!=1&&m!=2)||(packed!=0&&packed!=2))throw std::invalid_argument("3233 bounded replay geometry");
  auto&r=*c->runtime;r.assertOwnerThread();if(r.isGraphRecording())throw std::logic_error("3233 replay graph");
  infini::rt::DeviceGuard guard{r.device().type(),r.device().index(),true};
  if(cudaStreamSynchronize(static_cast<cudaStream_t>(input_stream))!=cudaSuccess)throw std::runtime_error("3233 source stream sync");
  a=rt::Tensor::from_blob(const_cast<void*>(x),{m,1536},rt::DataType::kBFloat16,r.device());
  b=rt::Tensor::from_blob(const_cast<void*>(w),{7168,1536},rt::DataType::kBFloat16,r.device());
  out=rt::Tensor::from_blob(y,{m,7168},rt::DataType::kBFloat16,r.device());
  if(packed==0){
    infinilm::layers::linear::dense_projection(r,a,b,std::optional<rt::Tensor>{},1.f,std::nullopt,false,out);
  }else{
    for(size_t i=0;i<m;++i){
      auto aa=rt::Tensor::from_blob(const_cast<char*>(static_cast<const char*>(x))+i*1536*2,{1,1536},rt::DataType::kBFloat16,r.device());
      auto yy=rt::Tensor::from_blob(static_cast<char*>(y)+i*7168*2,{1,7168},rt::DataType::kBFloat16,r.device());
      infinilm::layers::linear::dense_projection(r,aa,b,std::optional<rt::Tensor>{},1.f,std::nullopt,false,yy);
    }
  }
  r.syncStream();return 0;
 }catch(const std::exception&e){error=e.what();if(c)try{c->runtime->syncStream();}catch(...){std::terminate();}return 1;}
}
#include "engine/distributed/kimi_reduce_precision_r3226.hpp"

extern "C" int kimi_projection_r3883_frame(void*raw,size_t requests,int end){
 if(!raw)return 1;auto*c=static_cast<Context*>(raw);
 return end?modern_kimi_reduce_end_r3226(c->runtime.get()):modern_kimi_reduce_begin_r3226(c->runtime.get(),requests);
}
