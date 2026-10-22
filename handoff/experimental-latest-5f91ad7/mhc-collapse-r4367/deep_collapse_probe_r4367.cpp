#include "runtime/runtime.hpp"
#include "runtime/tensor.hpp"
#include "execution/submit.hpp"
#include <infini/rt/device_guard.h>
#include <nlohmann/json.hpp>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <thread>
#include <exception>
#include <cstring>
namespace rt=infinilm::runtime;namespace ex=infinilm::execution;
extern "C" int modern_deep_collapse_r4367(const float*,const float*,void*,size_t,int,int,void*);
extern "C" int modern_deepseek_post_compare_r135(const void*,const void*,size_t,int,void*,void*);
namespace {thread_local std::string result;
void check(int s){if(s)throw std::runtime_error(cudaGetErrorString(static_cast<cudaError_t>(s)));}
void*ptr(const nlohmann::json&x){return reinterpret_cast<void*>(x.get<uintptr_t>());}
double median(std::vector<double>x){std::sort(x.begin(),x.end());return(x[x.size()/2]+x[(x.size()-1)/2])/2;}
}
extern "C" int modern_deep_collapse_probe_r4367(const char*payload,int workers){
 nlohmann::json d={{"completed",false},{"scope","LatestSDK native Mul/Sum/Cast collapse tail vs FP32 fused three orders. Standalone component, no full mHC normalization/projection, weights, model IDs or throughput qualification."}};
 try{
  if(!payload||(workers!=1&&workers!=8))throw std::invalid_argument("4367 worker/payload contract");
  auto q=nlohmann::json::parse(payload);size_t T=q.at("tokens");bool low=q.at("bf16");
  if(!T||T>4096||q.at("original_out").size()!=workers||q.at("candidate_out").size()!=workers)throw std::invalid_argument("4367 geometry contract");
  std::vector<std::thread>ts;std::vector<std::exception_ptr>errors(workers);std::vector<nlohmann::json>reports(workers);
  for(int rank=0;rank<workers;++rank)ts.emplace_back([&,rank]{try{
   infini::rt::DeviceGuard guard{rt::Device::Type::kHygon,rank,true};auto owner=rt::Runtime::create({rt::Device::Type::kHygon,rank});auto&r=*owner;r.enableOwnerDrainFailStop();
   auto dtype=low?rt::DataType::kBFloat16:rt::DataType::kFloat32;
   auto xf=rt::Tensor::empty(r,{T,4,4096},rt::DataType::kFloat32),pre=rt::Tensor::empty(r,{T,4},rt::DataType::kFloat32),stats=rt::Tensor::empty(r,{16},rt::DataType::kUInt8);
   r.memcpyH2D(xf->data(),ptr(q.at("residual_fp32")),xf->nbytes(),false);r.memcpyH2D(pre->data(),ptr(q.at("pre")),pre->nbytes(),false);
   auto native=[&](){auto weighted=rt::Tensor::empty(r,{T,4,4096},rt::DataType::kFloat32);auto sum=rt::Tensor::empty(r,{T,4096},rt::DataType::kFloat32);
    ex::submit<infini::ops::Mul>(r,infini::ops::Mul::DefaultConfig(r.device().type()),pre->view({T,4,1}),xf,weighted);
    ex::submit<infini::ops::Sum>(r,infini::ops::Sum::DefaultConfig(r.device().type()),weighted,std::optional<std::vector<int64_t>>{{1}},false,std::optional<infini::ops::DataType>{},sum);
    auto out=sum;if(low){out=rt::Tensor::empty(r,{T,4096},dtype);ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),sum,out);}r.syncStream();weighted.reset();sum.reset();r.syncStream();return out;};
   auto fused=[&](int variant){auto out=rt::Tensor::empty(r,{T,4096},dtype);check(modern_deep_collapse_r4367(reinterpret_cast<const float*>(xf->data()),reinterpret_cast<const float*>(pre->data()),out->data(),T,int(low),variant,r.stream()));r.syncStream();return out;};
   auto compare=[&](const rt::Tensor&a,const rt::Tensor&b){check(modern_deepseek_post_compare_r135(a->data(),b->data(),a->numel(),int(low),stats->data(),r.stream()));r.syncStream();struct S{unsigned long long changed;unsigned finite,unused;}s{};r.memcpyD2H(&s,stats->data(),16);if(s.finite)throw std::runtime_error("4367 nonfinite output");return s.changed;};
   auto ref=native();nlohmann::json row={{"rank",rank},{"variants",nlohmann::json::array()}};
   for(int v=0;v<3;++v){auto out=fused(v);auto changed=compare(out,ref);auto again=fused(v);auto repeat=compare(out,again);row["variants"].push_back({{"variant",v},{"different_elements",changed},{"repeat_differences",repeat}});if(v==0)r.memcpyD2H(ptr(q.at("candidate_out").at(rank)),out->data(),out->nbytes());}
   r.memcpyD2H(ptr(q.at("original_out").at(rank)),ref->data(),ref->nbytes());ref.reset();
   std::vector<double>times[2];rt::Tensor output;
   for(int rep=0;rep<12;++rep)for(int j=0;j<2;++j){int mode=rep%2?1-j:j;auto start=std::chrono::steady_clock::now();output=mode?fused(0):native();r.syncStream();double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();if(rep>=2)times[mode].push_back(ms);output.reset();}
   row["native_wall_ms"]=median(times[0]);row["fused_wall_ms"]=median(times[1]);row["native_samples_ms"]=times[0];row["fused_samples_ms"]=times[1];
   stats.reset();pre.reset();xf.reset();r.syncStream();r.trimMemory();owner.reset();row["owner_shutdown_completed"]=true;reports[rank]=row;
  }catch(...){errors[rank]=std::current_exception();}});
  for(auto&t:ts)t.join();for(auto&e:errors)if(e)std::rethrow_exception(e);d["reports"]=reports;d["completed"]=true;result=d.dump();return 0;
 }catch(const std::exception&e){d["error"]=e.what();result=d.dump();return 1;}
}
extern "C" const char*modern_deep_collapse_result_r4367(){return result.c_str();}
