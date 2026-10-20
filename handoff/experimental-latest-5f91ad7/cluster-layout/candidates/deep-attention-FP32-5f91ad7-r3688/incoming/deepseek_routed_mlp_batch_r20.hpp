#pragma once
#include "runtime/runtime.hpp"
#include "execution/submit.hpp"
#include <optional>
#include <vector>
#include <stdexcept>
extern "C" int deep_routed_validate_r20(const int64_t*,size_t,size_t,int*,void*);
extern "C" int deep_routed_dot_r20(int,int,void*,const void*,const void*,const void*,const void*,const void*,size_t,void*);
namespace infinilm::models::deepseek_v4::routed_r20 {
namespace rt=infinilm::runtime;namespace ex=infinilm::execution;
struct Work{
 rt::Runtime&r;std::vector<rt::Tensor> held;
 ~Work(){try{r.syncStream();}catch(...){std::terminate();}}
 rt::Tensor make(rt::Shape shape,rt::DataType dtype){auto t=rt::Tensor::empty(r,shape,dtype);held.push_back(t);return t;}
};
inline rt::Tensor pack(Work&w,const rt::Tensor&x,rt::Tensor&scales){
 auto packed=w.make(x->shape(),rt::DataType::kInt8);scales=w.make({x->size(0),1},rt::DataType::kFloat32);
 auto config=infini::ops::PerChannelQuantI8::DefaultConfig(w.r.device().type());config.set_implementation_index(0);
 ex::submit<infini::ops::PerChannelQuantI8>(w.r,ex::RecordingPolicy::HostOnly,config,x,packed,scales);return packed;
}
inline rt::Tensor forward(rt::Runtime&r,const rt::Tensor&x,const rt::Tensor&ids,const rt::Tensor&weights,
 const rt::Tensor&w13,const rt::Tensor&s13,const rt::Tensor&w2,const rt::Tensor&s2){
 r.assertOwnerThread();Work work{r};size_t tokens=x->size(0),routes=tokens*6;
 auto status=work.make({1},rt::DataType::kInt32);r.setDeviceMemory(status->data(),0,4);
 if(deep_routed_validate_r20(reinterpret_cast<const int64_t*>(ids->data()),routes,256,reinterpret_cast<int*>(status->data()),r.stream()))throw std::runtime_error("DeepSeek routed validation launch failed");
 int invalid=0;r.memcpyD2H(&invalid,status->data(),4);if(invalid)throw std::invalid_argument("DeepSeek selected expert outside bank");
 rt::Tensor scales;auto packed=pack(work,x,scales);auto gu=work.make({routes,512},rt::DataType::kBFloat16);
 if(deep_routed_dot_r20(1,64,gu->data(),packed->data(),scales->data(),ids->data(),w13->data(),s13->data(),routes,r.stream()))throw std::runtime_error("DeepSeek routed gate/up launch failed");
 auto clamped=work.make({routes,512},rt::DataType::kBFloat16);
 ex::submit<infini::ops::Clamp>(r,infini::ops::Clamp::DefaultConfig(r.device().type()),gu->narrow({{1,0,256}}),std::optional<double>{},std::optional<double>{10},clamped->narrow({{1,0,256}}));
 ex::submit<infini::ops::Clamp>(r,infini::ops::Clamp::DefaultConfig(r.device().type()),gu->narrow({{1,256,256}}),std::optional<double>{-10},std::optional<double>{10},clamped->narrow({{1,256,256}}));
 auto activated=work.make({routes,256},rt::DataType::kBFloat16);
 ex::submit<infini::ops::SiluAndMul>(r,infini::ops::SiluAndMul::DefaultConfig(r.device().type()),clamped,activated);
 rt::Tensor down_scales;auto down_input=pack(work,activated,down_scales);auto local=work.make({routes,4096},rt::DataType::kBFloat16);
 if(deep_routed_dot_r20(0,32,local->data(),down_input->data(),down_scales->data(),ids->data(),w2->data(),s2->data(),routes,r.stream()))throw std::runtime_error("DeepSeek routed down launch failed");
 auto f=work.make({routes,4096},rt::DataType::kFloat32),weighted=work.make({routes,4096},rt::DataType::kFloat32),out=work.make({tokens,4096},rt::DataType::kFloat32);
 ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),local,f);
 ex::submit<infini::ops::Mul>(r,infini::ops::Mul::DefaultConfig(r.device().type()),f,weights->view({routes,1}),weighted);
 ex::submit<infini::ops::Sum>(r,infini::ops::Sum::DefaultConfig(r.device().type()),weighted->view({tokens,6,4096}),std::optional<std::vector<int64_t>>{{1}},false,std::optional<infini::ops::DataType>{},out);
 r.syncStream();return out;
}
}
