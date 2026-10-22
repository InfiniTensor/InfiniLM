#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
namespace {
__device__ void store_r4367(void*out,size_t i,float value,int low){
 if(!low){static_cast<float*>(out)[i]=value;return;}
 union{unsigned u;float f;}v;v.f=value;
 unsigned rounded=v.u+0x7fff+((v.u>>16)&1);
 if((v.u&0x7fffffff)>0x7f800000)rounded=v.u|0x00400000;
 static_cast<uint16_t*>(out)[i]=uint16_t(rounded>>16);
}
__global__ void collapse_r4367(const float*x,const float*pre,void*out,size_t tokens,int low,int variant){
 size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=tokens*4096)return;
 size_t t=i/4096,h=i%4096;const float*p=x+t*16384+h;const float*w=pre+t*4;
 float a=p[0]*w[0],b=p[4096]*w[1],c=p[8192]*w[2],d=p[12288]*w[3];
 float sum=variant==0?((a+b)+c)+d:variant==1?(a+b)+(c+d):(a+c)+(b+d);
 store_r4367(out,i,sum,low);
}
}
extern "C" int modern_deep_collapse_r4367(const float*x,const float*pre,void*out,size_t tokens,int low,int variant,void*stream){
 if(!x||!pre||!out||!tokens||tokens>4096||(low!=0&&low!=1)||variant<0||variant>2)return int(cudaErrorInvalidValue);
 collapse_r4367<<<(tokens*4096+255)/256,256,0,static_cast<cudaStream_t>(stream)>>>(x,pre,out,tokens,low,variant);
 return int(cudaGetLastError());
}
