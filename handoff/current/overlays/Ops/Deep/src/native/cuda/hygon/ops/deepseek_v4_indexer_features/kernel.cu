#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <stdexcept>
namespace infini::ops::dsv4_index_features_detail {
__device__ unsigned char encode(float x){unsigned sign=signbit(x)?128:0;float v=fminf(fabsf(x),448.f);unsigned code;if(v<0x1p-6f)code=unsigned(__float2int_rn(v*512.f));else{int e=int((__float_as_uint(v)>>23)&255)-127;int m=__float2int_rn(ldexpf(v,3-e))-8;code=unsigned(((e+7)<<3)+m);}return static_cast<unsigned char>(sign|code);}
__global__ void transform(const __nv_bfloat16*input,const __nv_bfloat16*norm,const float*rope,unsigned char*packed,float*scales,int*status,size_t heads,bool normalize,float eps){
 size_t row=blockIdx.x,lane=threadIdx.x,token=row/heads;__shared__ float values[128],scratch[128];__shared__ int invalid;if(lane==0)invalid=0;__syncthreads();float x=float(input[row*128+lane]);if(!isfinite(x)||(normalize&&!isfinite(float(norm[lane])))||(lane<64&&!isfinite(rope[token*64+lane])))atomicOr(&invalid,1);__syncthreads();if(invalid){if(lane==0)atomicOr(status,2);return;}
 if(normalize){scratch[lane]=__fmul_rn(x,x);__syncthreads();for(unsigned d=64;d;d>>=1){if(lane<d)scratch[lane]=__fadd_rn(scratch[lane],scratch[lane+d]);__syncthreads();}float inv=rsqrtf(__fadd_rn(__fmul_rn(scratch[0],1.f/128.f),eps));x=__fmul_rn(__fmul_rn(x,inv),float(norm[lane]));}
 values[lane]=x;__syncthreads();if(lane>=64){unsigned pair=unsigned(lane-64)/2;float c=rope[token*64+pair*2],s=rope[token*64+pair*2+1],a=values[64+pair*2],b=values[64+pair*2+1];x=(lane&1)?__fadd_rn(__fmul_rn(b,c),__fmul_rn(a,s)):__fsub_rn(__fmul_rn(a,c),__fmul_rn(b,s));}__syncthreads();values[lane]=x;__syncthreads();
 for(unsigned span=1;span<128;span<<=1){float a=values[lane],b=values[lane^span];__syncthreads();values[lane]=(lane&span)?__fsub_rn(b,a):__fadd_rn(a,b);__syncthreads();}
 x=__fmul_rn(values[lane],0.08838834764831844055f);if(!isfinite(x))atomicOr(&invalid,1);scratch[lane]=fabsf(x);__syncthreads();for(unsigned d=64;d;d>>=1){if(lane<d)scratch[lane]=fmaxf(scratch[lane],scratch[lane+d]);__syncthreads();}if(invalid){if(lane==0)atomicOr(status,2);return;}
 float scale=__fmul_rn(fmaxf(scratch[0],1e-4f),1.f/448.f);packed[row*128+lane]=encode(__fdiv_rn(x,scale));if(lane==0)scales[row]=scale;
}
__global__ void commit(unsigned char*out,float*scale,const unsigned char*packed,const float*scales,size_t rows,const int*status){if(*status)return;for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<rows*128;i+=size_t(gridDim.x)*blockDim.x)out[i]=packed[i];for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<rows;i+=size_t(gridDim.x)*blockDim.x)scale[i]=scales[i];}
void Launch(void*out,void*scale,const void*input,const void*norm,const void*rope,void*status,void*workspace,size_t tokens,size_t heads,bool normalize,float eps,cudaStream_t stream){auto check=[](cudaError_t x){if(x!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(x));};size_t rows=tokens*heads;auto packed=static_cast<unsigned char*>(workspace);auto scales=reinterpret_cast<float*>(packed+rows*128);auto flag=static_cast<int*>(status);check(cudaMemsetAsync(flag,0,4,stream));transform<<<unsigned(rows),128,0,stream>>>(static_cast<const __nv_bfloat16*>(input),static_cast<const __nv_bfloat16*>(norm),static_cast<const float*>(rope),packed,scales,flag,heads,normalize,eps);check(cudaGetLastError());commit<<<unsigned(std::min<size_t>((rows*128+255)/256,1024)),256,0,stream>>>(static_cast<unsigned char*>(out),static_cast<float*>(scale),packed,scales,rows,flag);check(cudaGetLastError());}
}
