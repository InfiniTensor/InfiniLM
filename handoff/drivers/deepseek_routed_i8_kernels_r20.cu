#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {
__global__ void validate(const int64_t* ids,size_t count,size_t experts,int* status){
 size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
 if(i<count&&(ids[i]<0||size_t(ids[i])>=experts))atomicExch(status,1);
}
template<int W,bool Gate> __global__ void routed_dot(__nv_bfloat16* output,const int8_t* x,
 const float* xs,const int64_t* ids,const int8_t* weights,const float* scales,size_t routes){
 constexpr size_t N=Gate?512:4096,K=Gate?4096:256;
 size_t row=size_t(blockIdx.x)*(256/W)+threadIdx.x/W,route=blockIdx.y;
 if(row>=N||route>=routes)return;
 unsigned lane=threadIdx.x%W;size_t expert=size_t(ids[route]),source=Gate?route/6:route;
 const auto* a=reinterpret_cast<const int*>(x+source*K);
 const auto* b=reinterpret_cast<const int*>(weights+(expert*N+row)*K);
 int total=0;
 for(size_t k=lane;k<K/4;k+=W)total=__dp4a(a[k],b[k],total);
 #pragma unroll
 for(int delta=W/2;delta;delta/=2){
#if defined(__HIPCC__)
  total+=__shfl_down(total,delta,W);
#else
  total+=__shfl_down_sync(0xffffffffu,total,delta,W);
#endif
 }
 if(lane==0){
  // Original I8Gemm epilogue: multiply scales first, then the exact INT32 dot.
  float value=(xs[source]*scales[expert*N+row])*float(total);
  output[route*N+row]=static_cast<__nv_bfloat16>(value);
 }
}
}
extern "C" int deep_routed_validate_r20(const int64_t* ids,size_t count,size_t experts,int* status,void* stream){
 validate<<<unsigned((count+255)/256),256,0,static_cast<cudaStream_t>(stream)>>>(ids,count,experts,status);
 return int(cudaGetLastError());
}
extern "C" int deep_routed_dot_r20(int gate,int width,void* output,const void* x,const void* xs,
 const void* ids,const void* weights,const void* scales,size_t routes,void* stream){
 auto s=static_cast<cudaStream_t>(stream);
 auto out=static_cast<__nv_bfloat16*>(output);auto a=static_cast<const int8_t*>(x);
 auto as=static_cast<const float*>(xs);auto index=static_cast<const int64_t*>(ids);
 auto w=static_cast<const int8_t*>(weights);auto ws=static_cast<const float*>(scales);
 if(gate){
  if(width==64)routed_dot<64,true><<<dim3(128,unsigned(routes)),256,0,s>>>(out,a,as,index,w,ws,routes);
  else routed_dot<32,true><<<dim3(64,unsigned(routes)),256,0,s>>>(out,a,as,index,w,ws,routes);
 }else routed_dot<32,false><<<dim3(512,unsigned(routes)),256,0,s>>>(out,a,as,index,w,ws,routes);
 return int(cudaGetLastError());
}
