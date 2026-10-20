#include "native/cuda/nvidia/ops/i8_gemm/kernel.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <algorithm>
namespace infini::ops::nvidia_i8_gemm_detail {
template <typename T> __global__ void Post(T* c,const std::int32_t* accum,
    const float* as,const float* bs,const T* bias,Layout layout) {
  for(std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
      i<layout.m*layout.n;i+=std::size_t(gridDim.x)*blockDim.x) {
    const auto row=i/layout.n,col=i%layout.n;
    // Preserve the original postSym ordering and round output only once.
    float value=(as[row*layout.sar]*bs[col*layout.sbr])*float(accum[i]);
    if(bias) value=value+float(bias[row*layout.biasr+col*layout.biasc]);
    c[row*layout.cr+col*layout.cc]=T(value);
  }
}
void Epilogue(void* c,const std::int32_t* accum,const float* as,const float* bs,
              const void* bias,DataType dtype,Layout layout,cudaStream_t stream) {
  const unsigned blocks=unsigned(std::min<std::size_t>((layout.m*layout.n-1)/256+1,65535));
  switch(dtype) {
    case DataType::kFloat16:Post<<<blocks,256,0,stream>>>(static_cast<__half*>(c),accum,as,bs,static_cast<const __half*>(bias),layout);break;
    case DataType::kBFloat16:Post<<<blocks,256,0,stream>>>(static_cast<__nv_bfloat16*>(c),accum,as,bs,static_cast<const __nv_bfloat16*>(bias),layout);break;
    case DataType::kFloat32:Post<<<blocks,256,0,stream>>>(static_cast<float*>(c),accum,as,bs,static_cast<const float*>(bias),layout);break;
    default:throw std::invalid_argument("unsupported I8Gemm output dtype");
  }
  const auto error=cudaGetLastError();if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
}
