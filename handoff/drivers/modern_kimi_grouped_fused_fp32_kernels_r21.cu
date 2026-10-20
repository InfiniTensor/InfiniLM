#include "modern_kimi_grouped_batched_kernels.cu"
__global__ void decode_batch_f32_r21(const unsigned char*packed,const unsigned char*scales,const int*experts,float*out,size_t count,size_t rows,size_t columns){
 size_t per=rows*(columns/2);
 for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count*per;i+=size_t(gridDim.x)*blockDim.x){size_t local=i/per,within=i%per,source=size_t(experts[local])*per+within;float a,b;mxfp4DecodePair(packed[source],scales[source/16],a,b);out[2*i]=__bfloat162float(__float2bfloat16_rn(a));out[2*i+1]=__bfloat162float(__float2bfloat16_rn(b));}
}
__global__ void gather_f32_r21(const __nv_bfloat16*x,const long long*rows,float*out,size_t count,size_t hidden){
 for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count*hidden;i+=size_t(gridDim.x)*blockDim.x)out[i]=__bfloat162float(x[size_t(rows[i/hidden])*hidden+i%hidden]);
}
__global__ void activate_f32_r21(const float*gu,float*out,size_t rows,size_t intermediate){
 for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<rows*intermediate;i+=size_t(gridDim.x)*blockDim.x){size_t row=i/intermediate,col=i%intermediate;float g=gu[row*2*intermediate+col],u=gu[row*2*intermediate+intermediate+col];out[i]=__bfloat162float(__float2bfloat16_rn((4.f*tanhf(g/4.f)/(1.f+expf(-g)))*(25.f*tanhf(u/25.f))));}
}
extern "C" int modern_grouped_decode_batch_f32_r21(const void*p,const void*s,const void*ids,void*out,size_t count,size_t rows,size_t columns,void*stream){decode_batch_f32_r21<<<blocks(count*rows*(columns/2)),256,0,static_cast<cudaStream_t>(stream)>>>(static_cast<const unsigned char*>(p),static_cast<const unsigned char*>(s),static_cast<const int*>(ids),static_cast<float*>(out),count,rows,columns);return int(cudaGetLastError());}
extern "C" int modern_grouped_gather_f32_r21(const void*x,const void*ids,void*out,size_t rows,size_t hidden,void*stream){gather_f32_r21<<<blocks(rows*hidden),256,0,static_cast<cudaStream_t>(stream)>>>(static_cast<const __nv_bfloat16*>(x),static_cast<const long long*>(ids),static_cast<float*>(out),rows,hidden);return int(cudaGetLastError());}
extern "C" int modern_grouped_activate_f32_r21(const void*gu,void*out,size_t rows,size_t intermediate,void*stream){activate_f32_r21<<<blocks(rows*intermediate),256,0,static_cast<cudaStream_t>(stream)>>>(static_cast<const float*>(gu),static_cast<float*>(out),rows,intermediate);return int(cudaGetLastError());}
