#pragma once
namespace infini::ops::mxfp4_detail {
// Only thread0 consumes the result in these W13 kernels. Preserve the exact
// original256-lane binary tree while replacing later shared-memory rounds
// with intra-wave permutations. Other threads need no broadcast of the sum.
template<size_t N,int Wave>
__device__ __forceinline__ void reduceWaveR1044(float(&values)[N],float*scratch){
 size_t lane=threadIdx.x;
 #pragma unroll
 for(size_t i=0;i<N;++i)scratch[i*256+lane]=values[i];
 __syncthreads();
 if(lane<Wave){
  #pragma unroll
  for(size_t i=0;i<N;++i){
   const float*p=scratch+i*256;float a=p[lane]+p[lane+128],b=p[lane+64]+p[lane+192];values[i]=a+b;
   if constexpr(Wave==32){float c=p[lane+32]+p[lane+160],d=p[lane+96]+p[lane+224];values[i]+=c+d;}
  }
  for(int shift=Wave/2;shift>0;shift/=2){
   #pragma unroll
   for(size_t i=0;i<N;++i){
    float other;
    if constexpr(Wave==64)other=__uint_as_float(__builtin_amdgcn_ds_bpermute(unsigned((lane+shift)&63)*4,__float_as_uint(values[i])));
    else{
#if defined(__HIPCC__)
     other=__shfl_down(values[i],shift,32);
#else
     other=__shfl_down_sync(0xffffffffu,values[i],shift,32);
#endif
    }
    values[i]+=other;
   }
  }
 }
}
template<size_t N,int Wave>
__global__ void reduceProofR1044(const float*input,float*output){
 float values[N];for(size_t i=0;i<N;++i)values[i]=input[(size_t(blockIdx.x)*N+i)*256+threadIdx.x];extern __shared__ float scratch[];
 if constexpr(Wave==0)mxfp4BlockReduce(values,scratch);else reduceWaveR1044<N,Wave>(values,scratch);
 if(threadIdx.x==0)for(size_t i=0;i<N;++i)output[size_t(blockIdx.x)*N+i]=values[i];
}
}
