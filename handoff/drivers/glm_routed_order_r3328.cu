#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

namespace {
__device__ float decode(uint8_t v) {
  unsigned e=(v>>3)&15,m=v&7;
  float x=e?(e==15&&m==7?nanf(""):ldexpf(1.f+float(m)/8,int(e)-7)):ldexpf(float(m),-9);
  return v&128?-x:x;
}
__global__ void validate(const int64_t* ids,size_t count,size_t experts,int* status){
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<count&&(ids[i]<0||size_t(ids[i])>=experts))atomicExch(status,1);
}
template<int W> __device__ float reduce(float* v){
  constexpr int V=128/W;
  #pragma unroll
  for(int stride=V/2;stride;stride/=2){
    #pragma unroll
    for(int j=0;j<V;++j)if(j<stride)v[j]=__fadd_rn(v[j],v[j+stride]);
  }
  float x=v[0];
  #pragma unroll
  for(int shift=W/2;shift;shift/=2){
#if defined(__HIPCC__)
    x=__fadd_rn(x,__shfl_down(x,shift,W));
#else
    x=__fadd_rn(x,__shfl_down_sync(0xffffffffu,x,shift,W));
#endif
  }
  return x;
}
__global__ void gate_up(__nv_bfloat16* gate,__nv_bfloat16* up,const uint8_t* x,const float* xs,
 const int64_t* ids,const uint8_t* wg,const float* sg,const uint8_t* wu,const float* su,
 size_t routes,size_t offset,const int64_t*order){
  constexpr int W=64,V=2;
  size_t row=size_t(blockIdx.x)*4+threadIdx.x/W,route=size_t(order[blockIdx.y]);
  if(row>=256||route>=routes)return;
  unsigned lane=threadIdx.x%W;size_t token=route/8,e=size_t(ids[route]);
  float g=0.f,u=0.f;
  for(size_t b=0;b<48;++b){
    float gv[V],uv[V];
    #pragma unroll
    for(int j=0;j<V;++j){
      size_t k=b*128+lane+size_t(j)*W;
      float a=decode(x[token*6144+k]);
      gv[j]=__fadd_rn(a*decode(wg[(e*256+row)*6144+k]),0.f);
      uv[j]=__fadd_rn(a*decode(wu[(e*256+row)*6144+k]),0.f);
    }
    float gd=reduce<W>(gv),ud=reduce<W>(uv);
    if(lane==0){
      size_t si=e*16*48+((offset+row)/128)*48+b;
      g+=gd*xs[token*48+b]*sg[si];
      u+=ud*xs[token*48+b]*su[si];
    }
  }
  if(lane==0){gate[route*256+row]=static_cast<__nv_bfloat16>(g);up[route*256+row]=static_cast<__nv_bfloat16>(u);}
}
__global__ void down(__nv_bfloat16* output,const uint8_t* x,const float* xs,const int64_t* ids,
 const uint8_t* weight,const float* scales,size_t routes,size_t offset,const int64_t*order){
  constexpr int W=32,V=4;
  size_t row=size_t(blockIdx.x)*8+threadIdx.x/W,route=size_t(order[blockIdx.y]);
  if(row>=6144||route>=routes)return;
  unsigned lane=threadIdx.x%W;size_t e=size_t(ids[route]);float result=0.f;
  for(size_t b=0;b<2;++b){
    float values[V];
    #pragma unroll
    for(int j=0;j<V;++j){
      size_t k=b*128+lane+size_t(j)*W;
      values[j]=__fadd_rn(decode(x[route*256+k])*decode(weight[(e*6144+row)*256+k]),0.f);
    }
    float dot=reduce<W>(values);
    if(lane==0)result+=dot*xs[route*2+b]*scales[e*48*16+(row/128)*16+offset/128+b];
  }
  if(lane==0)output[route*6144+row]=static_cast<__nv_bfloat16>(result);
}
}

// Stable expert buckets change launch order only. Every output retains its
// original route slot and exactly the original dot/reduction arithmetic.
namespace {
__global__ void route_order(const int64_t*ids,int64_t*order,size_t routes){
 __shared__ unsigned counts[256];unsigned expert=threadIdx.x,count=0;
 for(size_t i=0;i<routes;++i)count+=ids[i]==int64_t(expert);
 counts[expert]=count;__syncthreads();size_t offset=0;
 for(unsigned e=0;e<expert;++e)offset+=counts[e];
 for(size_t i=0;i<routes;++i)if(ids[i]==int64_t(expert))order[offset++]=int64_t(i);
}
}
extern "C" int glm_routed_order_r3328(const void*ids,void*order,size_t routes,void*stream){
 if(!ids||!order||!routes||routes>32768||routes%8||ids==order)return int(cudaErrorInvalidValue);
 route_order<<<1,256,0,static_cast<cudaStream_t>(stream)>>>(static_cast<const int64_t*>(ids),static_cast<int64_t*>(order),routes);
 return int(cudaGetLastError());
}
extern "C" int glm_routed_gate_r3328(void* gate,void* up,const void* x,const void* xs,const void* ids,
 const void* wg,const void* sg,const void* wu,const void* su,size_t routes,size_t offset,const void*order,void* stream){
 if(!order||!routes||routes>32768||routes%8||offset%256||offset>1792)return int(cudaErrorInvalidValue);
 gate_up<<<dim3(64,unsigned(routes)),256,0,static_cast<cudaStream_t>(stream)>>>(
 static_cast<__nv_bfloat16*>(gate),static_cast<__nv_bfloat16*>(up),static_cast<const uint8_t*>(x),static_cast<const float*>(xs),static_cast<const int64_t*>(ids),static_cast<const uint8_t*>(wg),static_cast<const float*>(sg),static_cast<const uint8_t*>(wu),static_cast<const float*>(su),routes,offset,static_cast<const int64_t*>(order));
 return int(cudaGetLastError());
}
extern "C" int glm_routed_down_r3328(void* output,const void* x,const void* xs,const void* ids,
 const void* weight,const void* scales,size_t routes,size_t offset,const void*order,void* stream){
 if(!order||!routes||routes>32768||routes%8||offset%256||offset>1792)return int(cudaErrorInvalidValue);
 down<<<dim3(768,unsigned(routes)),256,0,static_cast<cudaStream_t>(stream)>>>(
 static_cast<__nv_bfloat16*>(output),static_cast<const uint8_t*>(x),static_cast<const float*>(xs),static_cast<const int64_t*>(ids),static_cast<const uint8_t*>(weight),static_cast<const float*>(scales),routes,offset,static_cast<const int64_t*>(order));
 return int(cudaGetLastError());
}
