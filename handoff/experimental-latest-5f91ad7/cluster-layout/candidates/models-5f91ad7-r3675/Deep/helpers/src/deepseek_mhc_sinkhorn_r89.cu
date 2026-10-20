#include <cuda_runtime.h>
#include <cstddef>
#include <atomic>
namespace {std::atomic<size_t> calls_r89[8]{};}
extern "C" int modern_deepseek_sinkhorn_r89_counts(size_t*out){if(!out)return 1;for(size_t i=0;i<8;++i)out[i]=calls_r89[i].load();return 0;}
// One token's 4x4 matrix remains in registers for the original alternating
// column/row normalizations. Keep every FP32 division and epsilon addition.
__global__ void sinkhorn_r79(const float*input,float*output,size_t tokens,float eps,int iterations){
 size_t token=blockIdx.x*blockDim.x+threadIdx.x;if(token>=tokens)return;
 float a[16];for(int i=0;i<16;++i)a[i]=input[token*16+i];
 for(int iteration=0;iteration<iterations;++iteration){
  if(iteration){
   for(int row=0;row<4;++row){float sum=(a[4*row]+a[4*row+1])+(a[4*row+2]+a[4*row+3]);float denominator=sum+eps;for(int col=0;col<4;++col)a[4*row+col]=a[4*row+col]/denominator;}
  }
  for(int col=0;col<4;++col){float sum=((a[col]+a[4+col])+a[8+col])+a[12+col];float denominator=sum+eps;for(int row=0;row<4;++row)a[4*row+col]=a[4*row+col]/denominator;}
 }
 for(int i=0;i<16;++i)output[token*16+i]=a[i];
}
extern "C" int modern_deepseek_sinkhorn_r89(const float*input,float*output,size_t tokens,float eps,int iterations,void*stream,int device){
 if(!input||!output||!tokens||iterations<1||iterations>128||device<0||device>=8)return int(cudaErrorInvalidValue);
 ++calls_r89[device];
 sinkhorn_r79<<<(tokens+127)/128,128,0,static_cast<cudaStream_t>(stream)>>>(input,output,tokens,eps,iterations);
 return int(cudaGetLastError());
}
