#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <stdexcept>
namespace infini::ops::dsv4_cache_detail {
// DeepSeek V4 layout, FlashMLA ba89a3466e9470ad08ab39738d4e7bb66989e1e7:
// page: B rows [448 E4M3 +64 BF16], followed by B rows [7 E8M0 +padding].
__device__ unsigned char encode(float x){unsigned sign=signbit(x)?128:0;float v=fminf(fabsf(x),448.f);unsigned code;if(v<0x1p-6f)code=unsigned(__float2int_rn(v*512.f));else{int e=int((__float_as_uint(v)>>23)&255)-127;int m=__float2int_rn(ldexpf(v,3-e))-8;code=unsigned(((e+7)<<3)+m);}return static_cast<unsigned char>(sign|code);}
__device__ float decode(unsigned char b){unsigned e=(b>>3)&15,m=b&7;float v=e?(e==15&&m==7?nanf(""):ldexpf(1.f+float(m)/8,int(e)-7)):ldexpf(float(m),-9);return b&128?-v:v;}
__global__ void finite_input(const __nv_bfloat16*x,size_t n,int*flag){for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x)if(!isfinite(float(x[i])))atomicOr(flag,4);}
__global__ void store(const __nv_bfloat16*x,const int64_t*slots,unsigned char*cache,size_t block,size_t stride,const int*status){
 if(*status)return;
 size_t token=blockIdx.x,tile=blockIdx.y,lane=threadIdx.x,slot=size_t(slots[token]),page=slot/block,offset=slot%block;unsigned char*data=cache+page*stride+offset*576;unsigned char*scales=cache+page*stride+block*576+offset*8;
 __shared__ float maxima[64];float value=float(x[token*512+tile*64+lane]);maxima[lane]=fabsf(value);__syncthreads();for(unsigned step=32;step;step>>=1){if(lane<step)maxima[lane]=fmaxf(maxima[lane],maxima[lane+step]);__syncthreads();}
 float s=__fmul_rn(maxima[0],1.f/448.f);int exponent=int(ceilf(log2f(fmaxf(s,1e-4f))));float scale=ldexpf(1.f,exponent);if(lane==0)scales[tile]=static_cast<unsigned char>(exponent+127);data[tile*64+lane]=encode(__fdiv_rn(value,scale));
 if(tile==0)reinterpret_cast<unsigned short*>(data+448)[lane]=__bfloat16_as_ushort(x[token*512+448+lane]);
}
__global__ void gather(const unsigned char*cache,const int64_t*slots,__nv_bfloat16*out,size_t count,size_t block,size_t stride,const int*status){
 if(*status)return;
 for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count*512;i+=size_t(gridDim.x)*blockDim.x){size_t token=i/512,dim=i%512,slot=size_t(slots[token]),page=slot/block,offset=slot%block;const unsigned char*data=cache+page*stride+offset*576;
  if(dim>=448)out[i]=__ushort_as_bfloat16(reinterpret_cast<const unsigned short*>(data+448)[dim-448]);
  else{unsigned char exponent=cache[page*stride+block*576+offset*8+dim/64];out[i]=__float2bfloat16_rn(ldexpf(decode(data[dim]),int(exponent)-127));}
 }
}

__global__ void validate_slots(const int64_t*slots,size_t count,size_t capacity,unsigned int*seen,int*status,bool write){
 for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=size_t(gridDim.x)*blockDim.x){int64_t slot=slots[i];if(slot<0||size_t(slot)>=capacity){atomicOr(status,1);continue;}if(write){unsigned bit=1u<<(unsigned(slot)%32);unsigned old=atomicOr(seen+size_t(slot)/32,bit);if(old&bit)atomicOr(status,2);}}
}
void Launch(size_t mode,size_t count,size_t pages,size_t block,size_t stride,void*cache,const void*slots,void*values,void*status,void*bitmap,cudaStream_t stream){
 auto check=[](cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));};auto flag=static_cast<int*>(status);check(cudaMemsetAsync(flag,0,4,stream));
 if(mode==0)check(cudaMemsetAsync(bitmap,0,((pages*block+31)/32)*4,stream));
 unsigned slot_grid=unsigned(std::min<size_t>((count+255)/256,256)),value_grid=unsigned(std::min<size_t>((count*512+255)/256,256));
 validate_slots<<<slot_grid,256,0,stream>>>(static_cast<const int64_t*>(slots),count,pages*block,static_cast<unsigned int*>(bitmap),flag,mode==0);check(cudaGetLastError());
 if(mode==0){finite_input<<<value_grid,256,0,stream>>>(static_cast<const __nv_bfloat16*>(values),count*512,flag);check(cudaGetLastError());store<<<dim3(unsigned(count),7),64,0,stream>>>(static_cast<const __nv_bfloat16*>(values),static_cast<const int64_t*>(slots),static_cast<unsigned char*>(cache),block,stride,flag);}
 else gather<<<value_grid,256,0,stream>>>(static_cast<const unsigned char*>(cache),static_cast<const int64_t*>(slots),static_cast<__nv_bfloat16*>(values),count,block,stride,flag);
 check(cudaGetLastError());
}
}
