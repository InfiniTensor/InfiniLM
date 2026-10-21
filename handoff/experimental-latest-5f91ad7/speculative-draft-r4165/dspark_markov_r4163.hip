#include <hip/hip_runtime.h>
#include <stdint.h>
#include <math.h>
__device__ __forceinline__ float from_bf16(uint16_t x){union{uint32_t u;float f;}v;v.u=uint32_t(x)<<16;return v.f;}
__device__ __forceinline__ uint16_t to_bf16(float x){union{uint32_t u;float f;}v;v.f=x;v.u+=0x7fffu+((v.u>>16)&1u);return uint16_t(v.u>>16);}
template<int B>__global__ void markov(const int64_t*previous,const uint16_t*w1,const uint16_t*w2,const uint16_t*base,uint16_t*out,size_t stride){
 const int lane=threadIdx.x%64,row=blockIdx.x*4+threadIdx.x/64;const int V=163840,R=256;if(row>=V)return;float sum[B]{};int64_t ids[B];
 #pragma unroll
 for(int b=0;b<B;++b)ids[b]=previous[b];
 #pragma unroll
 for(int j=0;j<4;++j){int k=lane+j*64;float w=from_bf16(w2[size_t(row)*R+k]);
  #pragma unroll
  for(int b=0;b<B;++b){float x=ids[b]>=0&&ids[b]<V?from_bf16(w1[size_t(ids[b])*R+k]):NAN;sum[b]=fmaf(w,x,sum[b]);}
 }
 #pragma unroll
 for(int b=0;b<B;++b){for(int shift=32;shift;shift/=2)sum[b]+=__shfl_down(sum[b],shift,64);if(lane==0){float bias=from_bf16(to_bf16(sum[b]));out[size_t(b)*V+row]=to_bf16(from_bf16(base[size_t(b)*stride+row])+bias);}}
}
extern "C" int dspark_markov_logits_r4163(int device,int B,int V,int R,size_t stride,const void*previous,const void*w1,const void*w2,const void*base,void*out,void*stream){
 int actual=-1;if(hipGetDevice(&actual)!=hipSuccess||actual!=device||device<0||device>=8||V!=163840||R!=256||stride<size_t(V)||!previous||!w1||!w2||!base||!out||base==out)return 1;
 auto p=static_cast<const int64_t*>(previous);auto a=static_cast<const uint16_t*>(w1);auto b=static_cast<const uint16_t*>(w2);auto c=static_cast<const uint16_t*>(base);auto o=static_cast<uint16_t*>(out);auto s=static_cast<hipStream_t>(stream);
 switch(B){case 1:hipLaunchKernelGGL((markov<1>),dim3(40960),dim3(256),0,s,p,a,b,c,o,stride);break;case 2:hipLaunchKernelGGL((markov<2>),dim3(40960),dim3(256),0,s,p,a,b,c,o,stride);break;case 4:hipLaunchKernelGGL((markov<4>),dim3(40960),dim3(256),0,s,p,a,b,c,o,stride);break;case 8:hipLaunchKernelGGL((markov<8>),dim3(40960),dim3(256),0,s,p,a,b,c,o,stride);break;default:return 2;}
 return hipPeekAtLastError()==hipSuccess?0:3;
}
