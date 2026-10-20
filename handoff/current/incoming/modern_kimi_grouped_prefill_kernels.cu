#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "native/cuda/ops/fused_moe_mxfp4/mxfp4_math.cuh"
__global__ void decode(const unsigned char*p,const unsigned char*s,__nv_bfloat16*out,size_t rows,size_t columns){
 for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<rows*(columns/2);i+=size_t(gridDim.x)*blockDim.x){float a,b;mxfp4DecodePair(p[i],s[i/16],a,b);out[2*i]=__float2bfloat16_rn(a);out[2*i+1]=__float2bfloat16_rn(b);}
}
__global__ void activate(const float*gu,__nv_bfloat16*out,size_t rows,size_t intermediate){
 for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<rows*intermediate;i+=size_t(gridDim.x)*blockDim.x){size_t row=i/intermediate,col=i%intermediate;float g=gu[row*2*intermediate+col],u=gu[row*2*intermediate+intermediate+col];out[i]=__float2bfloat16_rn((4.f*tanhf(g/4.f)/(1.f+expf(-g)))*(25.f*tanhf(u/25.f)));}
}
__global__ void scatter(float*out,const float*x,const long long*routes,const float*weights,size_t rows,size_t hidden){
 for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<rows*hidden;i+=size_t(gridDim.x)*blockDim.x){size_t route=size_t(routes[i/hidden]);out[route*hidden+i%hidden]=x[i]*weights[route];}
}
unsigned blocks(size_t n){return unsigned(n/256+bool(n%256)>65535?65535:n/256+bool(n%256));}
extern "C" int modern_grouped_decode(const void*p,const void*s,void*out,size_t rows,size_t columns,void*stream){decode<<<blocks(rows*(columns/2)),256,0,static_cast<cudaStream_t>(stream)>>>(static_cast<const unsigned char*>(p),static_cast<const unsigned char*>(s),static_cast<__nv_bfloat16*>(out),rows,columns);return int(cudaGetLastError());}
extern "C" int modern_grouped_activate(const void*gu,void*out,size_t rows,size_t intermediate,void*stream){activate<<<blocks(rows*intermediate),256,0,static_cast<cudaStream_t>(stream)>>>(static_cast<const float*>(gu),static_cast<__nv_bfloat16*>(out),rows,intermediate);return int(cudaGetLastError());}
extern "C" int modern_grouped_scatter(void*out,const void*x,const void*routes,const void*weights,size_t rows,size_t hidden,void*stream){scatter<<<blocks(rows*hidden),256,0,static_cast<cudaStream_t>(stream)>>>(static_cast<float*>(out),static_cast<const float*>(x),static_cast<const long long*>(routes),static_cast<const float*>(weights),rows,hidden);return int(cudaGetLastError());}
