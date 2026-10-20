#include "native/cuda/hygon/ops/block_quant_f8_e4m3/kernel.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
namespace infini::ops::block_quant_f8_detail {
__device__ uint8_t Encode(float x){
  if(isnan(x))return 127;
  const unsigned sign=signbit(x)?128:0;float value=fminf(fabsf(x),448.f);unsigned code=0;
  if(value<0x1p-6f)code=unsigned(__float2int_rn(value*512.f));
  else {const int exponent=int((__float_as_uint(value)>>23)&255)-127;
    const int mantissa=__float2int_rn(ldexpf(value,3-exponent))-8;code=unsigned(((exponent+7)<<3)+mantissa);}
  return uint8_t(sign|code);
}
template<class T> __global__ void Kernel(const T* input,uint8_t* packed,float* scale,size_t columns,size_t blocks){
  const size_t row=blockIdx.x,block=blockIdx.y,column=block*128+threadIdx.x;
  __shared__ float maximum[128];__shared__ int nanflag[128];
  float value=column<columns?float(input[row*columns+column]):0.f;
  maximum[threadIdx.x]=fabsf(value);nanflag[threadIdx.x]=isnan(value);
  __syncthreads();
  for(unsigned step=64;step;step>>=1){if(threadIdx.x<step){maximum[threadIdx.x]=fmaxf(maximum[threadIdx.x],maximum[threadIdx.x+step]);nanflag[threadIdx.x]|=nanflag[threadIdx.x+step];}__syncthreads();}
  // Match Torch's tensor/scalar division lowering: a rounded FP32 reciprocal
  // constant then multiplication. One ULP here changes FP8 ties for BF16 input.
  const float s=nanflag[0]?nanf(""):__fmul_rn(fmaxf(maximum[0],1e-12f),1.f/448.f);
  if(threadIdx.x==0)scale[row*blocks+block]=s;
  if(column<columns)packed[row*columns+column]=Encode(__fdiv_rn(value,s));
}
void Launch(DataType dtype,const void* input,uint8_t* packed,float* scale,size_t rows,size_t columns,cudaStream_t stream){
  const size_t blocks=(columns+127)/128;dim3 grid(rows,blocks);
  switch(dtype){
   case DataType::kFloat32:Kernel<<<grid,128,0,stream>>>(static_cast<const float*>(input),packed,scale,columns,blocks);break;
   case DataType::kFloat16:Kernel<<<grid,128,0,stream>>>(static_cast<const half*>(input),packed,scale,columns,blocks);break;
   case DataType::kBFloat16:Kernel<<<grid,128,0,stream>>>(static_cast<const __nv_bfloat16*>(input),packed,scale,columns,blocks);break;
   default:throw std::invalid_argument("FP8 block quant dtype unsupported");}
  auto status=cudaGetLastError();if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
}
}
