#include "modern_kimi_grouped_prefill_kernels.cu"
__global__ void decode_batch(const unsigned char*packed,const unsigned char*scales,const int*experts,__nv_bfloat16*out,size_t count,size_t rows,size_t columns){
 size_t per=rows*(columns/2);
 for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count*per;i+=size_t(gridDim.x)*blockDim.x){size_t local=i/per,within=i%per,source=size_t(experts[local])*per+within;float a,b;mxfp4DecodePair(packed[source],scales[source/16],a,b);out[2*i]=__float2bfloat16_rn(a);out[2*i+1]=__float2bfloat16_rn(b);}
}
__global__ void scatter_batch(float*out,const float*x,const long long*routes,const float*weights,size_t rows,size_t hidden){
 for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<rows*hidden;i+=size_t(gridDim.x)*blockDim.x){long long route=routes[i/hidden];if(route>=0)out[size_t(route)*hidden+i%hidden]=x[i]*weights[route];}
}
extern "C" int modern_grouped_decode_batch(const void*p,const void*s,const void*ids,void*out,size_t count,size_t rows,size_t columns,void*stream){decode_batch<<<blocks(count*rows*(columns/2)),256,0,static_cast<cudaStream_t>(stream)>>>(static_cast<const unsigned char*>(p),static_cast<const unsigned char*>(s),static_cast<const int*>(ids),static_cast<__nv_bfloat16*>(out),count,rows,columns);return int(cudaGetLastError());}
extern "C" int modern_grouped_scatter_batch(void*out,const void*x,const void*routes,const void*weights,size_t rows,size_t hidden,void*stream){scatter_batch<<<blocks(rows*hidden),256,0,static_cast<cudaStream_t>(stream)>>>(static_cast<float*>(out),static_cast<const float*>(x),static_cast<const long long*>(routes),static_cast<const float*>(weights),rows,hidden);return int(cudaGetLastError());}
