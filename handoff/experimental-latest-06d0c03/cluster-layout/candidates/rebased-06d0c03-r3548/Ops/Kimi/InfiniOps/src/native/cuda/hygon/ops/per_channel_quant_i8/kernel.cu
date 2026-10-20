#include "native/cuda/hygon/ops/per_channel_quant_i8/kernel.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <stdexcept>
namespace infini::ops::hygon_per_channel_i8_detail {
template <typename T> __global__ void Kernel(const T* x,std::int8_t* p,float* scale,
                                            std::size_t rows,std::size_t k) {
  __shared__ float maximum[256];
  for(std::size_t row=blockIdx.x;row<rows;row+=gridDim.x) {
    float m=0;
    for(std::size_t c=threadIdx.x;c<k;c+=blockDim.x)
      m=fmaxf(m,fabsf(float(x[row*k+c])));
    maximum[threadIdx.x]=m; __syncthreads();
    for(unsigned step=128;step;step/=2) {
      if(threadIdx.x<step) maximum[threadIdx.x]=fmaxf(maximum[threadIdx.x],maximum[threadIdx.x+step]);
      __syncthreads();
    }
    const float s=fmaxf(maximum[0]/127.f,1e-8f);
    if(threadIdx.x==0) scale[row]=s;
    const float inverse=1.f/s;
    for(std::size_t c=threadIdx.x;c<k;c+=blockDim.x) {
      float v=float(x[row*k+c])*inverse;
      int q=v>=0?int(floorf(fabsf(v)+0.5f)):-int(floorf(fabsf(v)+0.5f));
      q=q>127?127:(q< -127?-127:q); p[row*k+c]=std::int8_t(q);
    }
    __syncthreads();
  }
}
void Launch(const void* x,std::int8_t* p,float* scale,DataType dtype,
            std::size_t rows,std::size_t k,cudaStream_t stream) {
  if(!rows) return;
  const unsigned blocks=unsigned(std::min<std::size_t>(rows,65535));
  switch(dtype) {
    case DataType::kFloat16: Kernel<<<blocks,256,0,stream>>>(static_cast<const __half*>(x),p,scale,rows,k);break;
    case DataType::kBFloat16: Kernel<<<blocks,256,0,stream>>>(static_cast<const __nv_bfloat16*>(x),p,scale,rows,k);break;
    case DataType::kFloat32: Kernel<<<blocks,256,0,stream>>>(static_cast<const float*>(x),p,scale,rows,k);break;
    default: throw std::invalid_argument("unsupported per-channel INT8 dtype");
  }
  const auto error=cudaGetLastError();
  if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
}
