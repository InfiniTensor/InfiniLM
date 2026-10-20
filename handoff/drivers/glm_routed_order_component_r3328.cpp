#include "runtime/runtime.hpp"
#include <infini/rt/device_guard.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <vector>
#include <string>
namespace rt=infinilm::runtime;
extern "C" int glm_routed_gate_r19(void*,void*,const void*,const void*,const void*,const void*,const void*,const void*,const void*,size_t,size_t,void*);
extern "C" int glm_routed_down_r19(void*,const void*,const void*,const void*,const void*,const void*,size_t,size_t,void*);
extern "C" int glm_routed_order_r3328(const void*,void*,size_t,void*);
extern "C" int glm_routed_gate_r3328(void*,void*,const void*,const void*,const void*,const void*,const void*,const void*,const void*,size_t,size_t,const void*,void*);
extern "C" int glm_routed_down_r3328(void*,const void*,const void*,const void*,const void*,const void*,size_t,size_t,const void*,void*);
namespace {thread_local std::string error;
void ck(int v){if(v)throw std::runtime_error("CUDA status "+std::to_string(v));}
struct Events{cudaEvent_t a{},b{};Events(){ck(cudaEventCreate(&a));try{ck(cudaEventCreate(&b));}catch(...){cudaEventDestroy(a);throw;}}~Events(){if(a)cudaEventDestroy(a);if(b)cudaEventDestroy(b);}};
struct Held{rt::Runtime&r;std::vector<rt::Tensor>t;~Held(){try{r.syncStream();}catch(...){std::terminate();}}};
}
extern "C" const char*glm_route_order_error_r3328(){return error.c_str();}
extern "C" int glm_route_order_component_r3328(int device,int mode,int op,size_t tokens,size_t offset,
 const void*x,const void*xs,const void*ids,const void*wg,const void*sg,const void*wu,const void*su,void*out,void*out_up,void*producer_stream,double*gpu_ms,int*closed,void*order_out){
 std::shared_ptr<rt::Runtime>owner;
 try{
  if(device<0||device>=8||mode<0||mode>1||op<0||op>1||!tokens||tokens>4096||offset%256||offset>1792||!x||!xs||!ids||!wg||!sg||!out||(!op&&(!wu||!su||!out_up))||!gpu_ms||!closed||!order_out)throw std::invalid_argument("route-order bounded TP8 geometry");
  infini::rt::DeviceGuard guard{rt::Device::Type::kHygon,device,true};ck(cudaStreamSynchronize(static_cast<cudaStream_t>(producer_stream)));
  owner=rt::Runtime::create({rt::Device::Type::kHygon,device});std::weak_ptr<rt::Runtime>weak=owner;auto&r=*owner;r.enableOwnerDrainFailStop();
  {
   size_t routes=tokens*8;std::vector<int64_t>host(routes);r.memcpyD2H(host.data(),ids,routes*8);
   for(auto e:host)if(e<0||e>=256)throw std::invalid_argument("expert outside 256 bank");
   Held held{r};held.t.push_back(rt::Tensor::empty(r,{routes},rt::DataType::kInt64));auto&order=held.t.back();Events timer;ck(cudaEventRecord(timer.a,static_cast<cudaStream_t>(r.stream())));
   if(mode){ck(glm_routed_order_r3328(ids,order->data(),routes,r.stream()));
    if(op)ck(glm_routed_down_r3328(out,x,xs,ids,wg,sg,routes,offset,order->data(),r.stream()));
    else ck(glm_routed_gate_r3328(out,out_up,x,xs,ids,wg,sg,wu,su,routes,offset,order->data(),r.stream()));
   }else{
    if(op)ck(glm_routed_down_r19(out,x,xs,ids,wg,sg,routes,offset,r.stream()));
    else ck(glm_routed_gate_r19(out,out_up,x,xs,ids,wg,sg,wu,su,routes,offset,r.stream()));
   }
   ck(cudaEventRecord(timer.b,static_cast<cudaStream_t>(r.stream())));r.syncStream();float ms=0;ck(cudaEventElapsedTime(&ms,timer.a,timer.b));
   if(mode)r.memcpyD2H(order_out,order->data(),routes*8);else for(size_t i=0;i<routes;++i)static_cast<int64_t*>(order_out)[i]=int64_t(i);
   *gpu_ms=ms;
  }
  r.trimMemory();owner.reset();if(!weak.expired())throw std::logic_error("Runtime retained");*closed=1;return 0;
 }catch(const std::exception&e){error=e.what();if(owner)try{owner->syncStream();}catch(...){std::terminate();}return 1;}
}
