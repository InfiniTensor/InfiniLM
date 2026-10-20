#include "execution/submit.hpp"
#include "base/causal_conv1d.h"
#include <infini/rt/device_guard.h>
#include <cstring>
#include <vector>
namespace rt=infinilm::runtime;namespace ex=infinilm::execution;
namespace{thread_local std::string error;uint16_t bf(float v){uint32_t b;std::memcpy(&b,&v,4);return uint16_t(b>>16);}}
extern "C" const char*kimi_conv_error_r3746(){return error.c_str();}
extern "C" int kimi_conv_smoke_r3746(int device){std::shared_ptr<rt::Runtime>owner;try{
 if(device<0||device>=8)throw std::invalid_argument("3746 device");infini::rt::DeviceGuard guard{rt::Device::Type::kHygon,device,true};owner=rt::Runtime::create({rt::Device::Type::kHygon,device});owner->enableOwnerDrainFailStop();auto&r=*owner;
 for(size_t R:{1ul,4ul,8ul})for(size_t W:{1ul,4ul,17ul})for(bool disjoint:{false,true}){
  const size_t C=4608,P=129,T=R*W;std::vector<uint16_t>x(T*C),pool(P*C*3,0),out(T*C),gold(T*C);std::vector<float>w(C*4);std::vector<int32_t>off(R+1),read(R),write(R);
  for(size_t c=0;c<C;++c)for(size_t k=0;k<4;++k)w[c*4+k]=float(k+1);
  for(size_t i=0;i<R;++i){off[i]=int32_t(i*W);read[i]=int32_t(i+1);write[i]=int32_t((disjoint?65:1)+i);for(size_t t=0;t<W;++t)for(size_t c=0;c<C;++c)x[(i*W+t)*C+c]=bf(float((c+t+i)%7+1));}off[R]=int32_t(T);
  for(size_t i=0;i<R;++i)for(size_t t=0;t<W;++t)for(size_t c=0;c<C;++c){float v=0;for(size_t k=0;k<4;++k)if(t+k>=3){size_t j=t+k-3;v+=w[c*4+k]*float((c+j+i)%7+1);}gold[(i*W+t)*C+c]=bf(v);}
  auto q=rt::Tensor::empty(r,{1,T,C},rt::DataType::kBFloat16),state=rt::Tensor::empty(r,{P,C,3},rt::DataType::kBFloat16),weight=rt::Tensor::empty(r,{C,1,4},rt::DataType::kFloat32),y=rt::Tensor::empty(r,{1,T,C},rt::DataType::kBFloat16),offsets=rt::Tensor::empty(r,{R+1},rt::DataType::kInt32),reads=rt::Tensor::empty(r,{R},rt::DataType::kInt32),writes=rt::Tensor::empty(r,{R},rt::DataType::kInt32);
  r.memcpyH2D(q->data(),x.data(),x.size()*2);r.memcpyH2D(state->data(),pool.data(),pool.size()*2);r.memcpyH2D(weight->data(),w.data(),w.size()*4);r.memcpyH2D(offsets->data(),off.data(),off.size()*4);r.memcpyH2D(reads->data(),read.data(),read.size()*4);r.memcpyH2D(writes->data(),write.data(),write.size()*4);
  std::optional<rt::Tensor>none;auto config=infini::ops::CausalConv1d::DefaultConfig(r.device().type());config.set_implementation_index(0);ex::submit<infini::ops::CausalConv1d>(r,config,q,weight,state,none,std::optional<rt::Tensor>{offsets},std::optional<rt::Tensor>{reads},std::optional<rt::Tensor>{writes},y,none);r.syncStream();r.memcpyD2H(out.data(),y->data(),out.size()*2);r.memcpyD2H(pool.data(),state->data(),pool.size()*2);r.syncStream();if(out!=gold)throw std::runtime_error("3746 BF16/F32weight packedconv full output mismatch");
  for(size_t p=0;p<P;++p)for(size_t c=0;c<C;++c)for(size_t k=0;k<3;++k){uint16_t expected=0;for(size_t i=0;i<R;++i)if(p==size_t(write[i])&&W+k>=3){size_t j=W+k-3;expected=x[(i*W+j)*C+c];}if(pool[(p*C+c)*3+k]!=expected)throw std::runtime_error("3746 full state/untouched readslot mismatch");}
 }
 r.syncStream();r.trimMemory();std::weak_ptr<rt::Runtime>weak=owner;owner.reset();if(!weak.expired())throw std::runtime_error("3746 Runtime retained");return 0;
}catch(const std::exception&e){error=e.what();if(owner)try{owner->syncStream();}catch(...){std::terminate();}return 1;}}
