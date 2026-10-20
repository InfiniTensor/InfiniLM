#include "native/cuda/hygon/ops/fp8_block_linear/kernel.h"
#include <cublas_v2.h>
#include <string>
#include <thread>
#include <vector>
namespace {
thread_local std::string error;
using B=__nv_bfloat16;
struct Guard{int previous{-1};Guard(int d){if(cudaGetDevice(&previous)!=cudaSuccess||cudaSetDevice(d)!=cudaSuccess)throw std::runtime_error("device guard failed");}~Guard(){if(previous>=0&&cudaSetDevice(previous)!=cudaSuccess)std::terminate();}};
void check(cudaError_t s){if(s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(s));}
void blas(cublasStatus_t s){if(s!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("cublas "+std::to_string(int(s)));}
struct Plan{int device{},dtype{};std::thread::id owner;cudaStream_t stream{};cublasHandle_t handle{};size_t m{},n{},k{},sr{},sc{},ro{},co{},first{},blocks{},chunk{};B*x{},*w{};float*partial{};};
void release(Plan*p){if(!p)return;check(cudaStreamSynchronize(p->stream));if(p->handle)blas(cublasDestroy(p->handle));if(p->x)check(cudaFree(p->x));if(p->w)check(cudaFree(p->w));if(p->partial)check(cudaFree(p->partial));delete p;}
__global__ void unpack(const uint8_t*src,B*dst,size_t rows,size_t k,size_t first,size_t offset){
 size_t row=size_t(blockIdx.x)*2+threadIdx.x/128,lane=threadIdx.x%128,block=blockIdx.y;if(row>=rows)return;size_t column=(first+block)*128+lane;float value=0;if(column>=offset&&column-offset<k)value=infini::ops::fp8_block_detail::Decode(src[row*k+column-offset]);dst[(block*rows+row)*128+lane]=__float2bfloat16_rn(value);
}
template<class T>__global__ void combine(T*out,const float*partial,const float*xs,const float*ws,size_t count,size_t n,size_t sc,size_t ro,size_t first,size_t blocks,size_t begin){
 size_t p=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(p>=count*n)return;size_t t=p/n,row=p%n;float result=0;
 for(size_t b=0;b<blocks;++b)result+=partial[b*count*n+p]*xs[(begin+t)*sc+first+b]*ws[((ro+row)/128)*sc+first+b];out[(begin+t)*n+row]=static_cast<T>(result);
}
}
extern "C" const char*glm_fp8_error_r3806(){return error.c_str();}
extern "C" void*glm_fp8_create_r3806(int device,int dtype,size_t m,size_t n,size_t k,size_t sr,size_t sc,size_t ro,size_t co,void*stream){Plan*p=nullptr;try{
 if(device<0||device>=8||dtype<0||dtype>2||!m||m>8192||!n||n>32768||!k||k>32768||!sr||sr>512||!sc||sc>512||ro>sr*128||n>sr*128-ro||co>sc*128||k>sc*128-co)throw std::invalid_argument("776 bounded geometry/scale coverage invalid");Guard guard(device);p=new Plan;p->device=device;p->dtype=dtype;p->owner=std::this_thread::get_id();p->stream=static_cast<cudaStream_t>(stream);p->m=m;p->n=n;p->k=k;p->sr=sr;p->sc=sc;p->ro=ro;p->co=co;p->first=co/128;p->blocks=(co+k+127)/128-p->first;p->chunk=std::min(m,size_t(256));blas(cublasCreate(&p->handle));blas(cublasSetStream(p->handle,p->stream));blas(cublasSetPointerMode(p->handle,CUBLAS_POINTER_MODE_HOST));blas(cublasSetAtomicsMode(p->handle,CUBLAS_ATOMICS_NOT_ALLOWED));blas(cublasSetMathMode(p->handle,CUBLAS_DEFAULT_MATH));check(cudaMalloc(&p->x,p->blocks*m*128*sizeof(B)));check(cudaMalloc(&p->w,p->blocks*n*128*sizeof(B)));check(cudaMalloc(&p->partial,p->blocks*p->chunk*n*sizeof(float)));return p;
 }catch(const std::exception&e){error=e.what();if(p)try{Guard guard(p->device);release(p);}catch(...){std::terminate();}return nullptr;}}
extern "C" int glm_fp8_run_r3806(void*value,int mode,void*out,const void*x,const void*xs,const void*w,const void*ws){try{
 if(!value)throw std::invalid_argument("776 missing plan");auto&p=*static_cast<Plan*>(value);if(p.owner!=std::this_thread::get_id()||mode<0||mode>1||!out||!x||!xs||!w||!ws)throw std::invalid_argument("776 owner/mode/pointer invalid");Guard guard(p.device);
 if(mode==0){
  auto launch=[&](auto*typed){infini::ops::fp8_block_detail::Linear<<<dim3(p.n,p.m),256,0,p.stream>>>(typed,static_cast<const uint8_t*>(x),static_cast<const float*>(xs),static_cast<const uint8_t*>(w),static_cast<const float*>(ws),p.m,p.n,p.k,p.sc,p.ro,p.co);};
  if(p.dtype==2)launch(static_cast<float*>(out));else if(p.dtype==1)launch(static_cast<B*>(out));else launch(static_cast<half*>(out));
 }else{
  unpack<<<dim3((p.m+1)/2,p.blocks),256,0,p.stream>>>(static_cast<const uint8_t*>(x),p.x,p.m,p.k,p.first,p.co);unpack<<<dim3((p.n+1)/2,p.blocks),256,0,p.stream>>>(static_cast<const uint8_t*>(w),p.w,p.n,p.k,p.first,p.co);const float alpha=1,beta=0;
  for(size_t begin=0;begin<p.m;begin+=p.chunk){size_t count=std::min(p.chunk,p.m-begin);blas(cublasGemmStridedBatchedEx(p.handle,CUBLAS_OP_T,CUBLAS_OP_N,int(p.n),int(count),128,&alpha,p.w,CUDA_R_16BF,128,static_cast<long long>(p.n)*128,p.x+begin*128,CUDA_R_16BF,128,static_cast<long long>(p.m)*128,&beta,p.partial,CUDA_R_32F,int(p.n),static_cast<long long>(count)*p.n,int(p.blocks),CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
   auto launch=[&](auto*typed){combine<<<unsigned((count*p.n+255)/256),256,0,p.stream>>>(typed,p.partial,static_cast<const float*>(xs),static_cast<const float*>(ws),count,p.n,p.sc,p.ro,p.first,p.blocks,begin);};if(p.dtype==2)launch(static_cast<float*>(out));else if(p.dtype==1)launch(static_cast<B*>(out));else launch(static_cast<half*>(out));
  }
 }check(cudaGetLastError());return 0;
 }catch(const std::exception&e){error=e.what();return 1;}}
extern "C" int glm_fp8_destroy_r3806(void*value){try{if(!value)throw std::invalid_argument("776 missing plan");auto*p=static_cast<Plan*>(value);if(p->owner!=std::this_thread::get_id())throw std::runtime_error("776 destroy owner differs");Guard guard(p->device);release(p);return 0;}catch(const std::exception&e){error=e.what();return 1;}}
namespace {__global__ void domain776(B*out){unsigned code=threadIdx.x;out[code]=__float2bfloat16_rn(infini::ops::fp8_block_detail::Decode(uint8_t(code)));}}
extern "C" int glm_fp8_domain_r3806(void*output,void*stream){domain776<<<1,256,0,static_cast<cudaStream_t>(stream)>>>(static_cast<B*>(output));return int(cudaGetLastError());}

// Capacity is only storage. Every math/stride/chunk uses the actual M.
extern "C" int glm_fp8_run_m_r3806(void*raw,size_t actual_m,void*out,const void*x,const void*xs,const void*w,const void*ws){try{
 auto*p=static_cast<Plan*>(raw);if(!p||p->owner!=std::this_thread::get_id()||actual_m<32||actual_m>p->m)throw std::invalid_argument("3806 actualM/owner bounds");Plan frame=*p;frame.m=actual_m;frame.chunk=std::min(actual_m,size_t(256));return glm_fp8_run_r3806(&frame,1,out,x,xs,w,ws);
 }catch(const std::exception&e){error=e.what();return 1;}}
