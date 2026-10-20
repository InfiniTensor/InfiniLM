#pragma once
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <malloc.h>
#include <thread>
#include <tuple>
#include <type_traits>
#include <typeinfo>
namespace infinilm::execution::detail {
template<class Key>class HostOpAllocationTraceR25 {
 bool enabled_;int device_;size_t before_;
 static size_t live(){
#if defined(__GLIBC__) && __GLIBC_PREREQ(2,33)
  const auto m=mallinfo2();return m.uordblks+m.hblkhd;
#else
  return 0;
#endif
 }
 template<class T>static void shape(char*buf,size_t&pos,const T&value){
  if constexpr(std::is_same_v<std::decay_t<T>,infini::ops::Tensor>){
   auto add=[&](const char*fmt,size_t n){if(pos>=4095)return;int count=std::snprintf(buf+pos,4096-pos,fmt,n);if(count>0)pos+=std::min<size_t>(count,4095-pos);};
   add(" tensor_rank=%zu[",value.shape().size());for(auto dim:value.shape())add("%zu,",size_t(dim));add("]%zu",size_t(value.dtype()));
  }
 }
public:
 explicit HostOpAllocationTraceR25(int device):enabled_(false),device_(device),before_(0){const char*p=std::getenv("INFINILM_HOST_OP_ALLOC_R24");enabled_=p&&*p=='1'&&device_==0;if(enabled_)before_=live();}
 template<class...Args>void emit(const Args&...args)const{
  if(!enabled_)return;const size_t after=live();const auto delta=static_cast<long long>(after)-static_cast<long long>(before_);if(delta<131072&&delta>-131072)return;
  char buf[4096];int count=std::snprintf(buf,sizeof(buf),"HOST_OP_ALLOC_R25 device=%d tid=%zu op=%s before=%zu after=%zu delta=%lld",device_,std::hash<std::thread::id>{}(std::this_thread::get_id()),typeid(Key).name(),before_,after,delta);size_t pos=count>0?std::min<size_t>(count,4095):0;(shape(buf,pos,args),...);std::fprintf(stderr,"%s\n",buf);
 }
};
}
