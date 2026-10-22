#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <cmath>
namespace {
__device__ float load_value(const void*p,size_t index,int bf16){
 if(!bf16)return static_cast<const float*>(p)[index];
 union{unsigned bits;float value;}v;v.bits=unsigned(static_cast<const uint16_t*>(p)[index])<<16;return v.value;
}
__device__ void store_value(void*p,size_t index,float value,int bf16){
 if(!bf16){static_cast<float*>(p)[index]=value;return;}
 union{unsigned bits;float value;}v;v.value=value;
 unsigned rounded=v.bits+0x7fff+((v.bits>>16)&1);if((v.bits&0x7fffffff)>0x7f800000)rounded=v.bits|0x00400000;
 static_cast<uint16_t*>(p)[index]=uint16_t(rounded>>16);
}
__global__ void post_kernel(const void*block,const void*residual,const float*post,const float*comb,void*out,size_t tokens,int bf16,int variant){
 size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(index>=tokens*4*4096)return;
 size_t token=index/(4*4096),destination=(index/4096)%4,h=index%4096;
 float a=comb[token*16+destination]*load_value(residual,token*4*4096+h,bf16);
 float b=comb[token*16+4+destination]*load_value(residual,token*4*4096+4096+h,bf16);
 float c=comb[token*16+8+destination]*load_value(residual,token*4*4096+8192+h,bf16);
 float d=comb[token*16+12+destination]*load_value(residual,token*4*4096+12288+h,bf16);
 float mixed=variant==0?((a+b)+c)+d:variant==1?(a+b)+(c+d):(a+c)+(b+d);
 float scaled=post[token*4+destination]*load_value(block,token*4096+h,bf16);
 store_value(out,index,scaled+mixed,bf16);
}
__global__ void compare_kernel(const void*a,const void*b,size_t n,int bf16,unsigned long long*changed,unsigned*nonfinite){
 size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
 bool different=bf16?static_cast<const uint16_t*>(a)[i]!=static_cast<const uint16_t*>(b)[i]:static_cast<const unsigned*>(a)[i]!=static_cast<const unsigned*>(b)[i];
 if(different)atomicAdd(changed,1ULL);
 if(!isfinite(load_value(a,i,bf16))||!isfinite(load_value(b,i,bf16)))atomicOr(nonfinite,1u);
}
}
extern "C" int modern_deepseek_post_r135(const void*block,const void*residual,const float*post,const float*comb,void*out,size_t tokens,int bf16,int variant,void*stream){
 if(!block||!residual||!post||!comb||!out||!tokens||tokens>4096||(bf16!=0&&bf16!=1)||variant<0||variant>2)return int(cudaErrorInvalidValue);
 post_kernel<<<(tokens*4*4096+255)/256,256,0,static_cast<cudaStream_t>(stream)>>>(block,residual,post,comb,out,tokens,bf16,variant);return int(cudaGetLastError());
}
extern "C" int modern_deepseek_post_compare_r135(const void*a,const void*b,size_t n,int bf16,void*stats,void*stream){
 if(!a||!b||!n||!stats)return int(cudaErrorInvalidValue);auto s=static_cast<cudaStream_t>(stream);
 auto status=cudaMemsetAsync(stats,0,16,s);if(status!=cudaSuccess)return int(status);
 compare_kernel<<<(n+255)/256,256,0,s>>>(a,b,n,bf16,static_cast<unsigned long long*>(stats),reinterpret_cast<unsigned*>(static_cast<unsigned char*>(stats)+8));return int(cudaGetLastError());
}
