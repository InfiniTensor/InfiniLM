#include "execution/submit.hpp"
#include <infini/rt/device_guard.h>
#include <array>
#include <cstring>
namespace rt=infinilm::runtime;namespace ex=infinilm::execution;
namespace {thread_local std::string smoke_error;}
extern "C" const char*kimi_sdk_smoke_error_r3714(){return smoke_error.c_str();}
extern "C" int kimi_sdk_smoke_r3714(int device){
 std::shared_ptr<rt::Runtime>owner;rt::Tensor input,indices,output;
 try{
  if(device<0||device>=8)throw std::invalid_argument("3714 device");
  infini::rt::DeviceGuard guard{rt::Device::Type::kHygon,device,true};owner=rt::Runtime::create({rt::Device::Type::kHygon,device});owner->enableOwnerDrainFailStop();auto&r=*owner;r.assertOwnerThread();
  std::array<uint16_t,128>x{};for(size_t i=0;i<x.size();++i){float v=float(i);uint32_t bits;std::memcpy(&bits,&v,4);x[i]=uint16_t(bits>>16);}
  const std::array<int64_t,2>ids{0,3};std::array<uint16_t,16>actual{};
  input=rt::Tensor::empty(r,{16,8},rt::DataType::kBFloat16);indices=rt::Tensor::empty(r,{2},rt::DataType::kInt64);output=rt::Tensor::empty(r,{2,8},rt::DataType::kBFloat16);r.memcpyH2D(input->data(),x.data(),sizeof(x));r.memcpyH2D(indices->data(),ids.data(),sizeof(ids));
  ex::submit<infini::ops::IndexSelect>(r,infini::ops::IndexSelect::DefaultConfig(r.device().type()),input,indices,int64_t{0},output);r.syncStream();r.memcpyD2H(actual.data(),output->data(),sizeof(actual));r.syncStream();
  for(size_t i=0;i<actual.size();++i)if(actual[i]!=x[(i/8?24:0)+i%8])throw std::runtime_error("3714 BF16 IndexSelect complete output mismatch");
  input={};indices={};output={};r.trimMemory();std::weak_ptr<rt::Runtime>weak=owner;owner.reset();if(!weak.expired())throw std::runtime_error("3714 Runtime retained");return 0;
 }catch(const std::exception&e){smoke_error=e.what();if(owner){try{owner->syncStream();}catch(...){std::terminate();}}return 1;}
}
