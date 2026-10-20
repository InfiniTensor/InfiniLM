#pragma once
// Private experiment: reproduce the original active128 reduction tree in
// 32/64 physical lanes. FP8 Decode and block scale/accumulation are unchanged.
namespace infini::ops::fp8_block_detail {
__device__ inline float DecodeBitsR17(uint8_t b){
 unsigned e=(b>>3)&15,m=b&7,sign=unsigned(b&128)<<24;
 if(e==15&&m==7)return Decode(b);
 if(e)return __uint_as_float(sign|((e+120)<<23)|(m<<20));
 return __uint_as_float(sign|__float_as_uint(float(m)*0.001953125f));
}

template<class T,int Width>
__global__ void LinearBitsR17(T*out,const uint8_t*x,const float*xs,
 const uint8_t*w,const float*ws,size_t m,size_t n,size_t k,size_t scale_columns,
 size_t row_offset,size_t column_offset){
 constexpr int Virtual=128/Width,Groups=256/Width;
 size_t row=size_t(blockIdx.x)*Groups+threadIdx.x/Width,token=blockIdx.y;
 if(row>=n||token>=m)return;
 unsigned lane=threadIdx.x%Width;float result=0.f;
 for(size_t block=column_offset/128;block<(column_offset+k+127)/128;++block){
  float values[Virtual]{};
  #pragma unroll
  for(int j=0;j<Virtual;++j){
   size_t global_column=block*128+lane+size_t(j)*Width;
   if(global_column>=column_offset&&global_column-column_offset<k){
    size_t column=global_column-column_offset;
    values[j]=DecodeBitsR17(x[token*k+column])*DecodeBitsR17(w[row*k+column]);
   }
   // Original stride128 combines each active lane with an explicit zero.
   values[j]=__fadd_rn(values[j],0.f);
  }
  #pragma unroll
  for(int stride=Virtual/2;stride;stride/=2){
   #pragma unroll
   for(int j=0;j<Virtual;++j)if(j<stride)values[j]=__fadd_rn(values[j],values[j+stride]);
  }
  float value=values[0];
  #pragma unroll
  for(int shift=Width/2;shift;shift/=2){
#if defined(__HIPCC__)
   value=__fadd_rn(value,__shfl_down(value,shift,Width));
#else
   value=__fadd_rn(value,__shfl_down_sync(0xffffffffu,value,shift,Width));
#endif
  }
  if(lane==0)result+=value*xs[token*scale_columns+block]*ws[((row_offset+row)/128)*scale_columns+block];
 }
 if(lane==0)out[token*n+row]=static_cast<T>(result);
}
}

namespace infini::ops {
template<int Width>class Fp8BitsR17 final:public Operator<Fp8BlockLinear,Device::Type::kHygon>{
public:
 using Parent=Operator<Fp8BlockLinear,Device::Type::kHygon>;
 using Parent::Parent;
 void operator()(Tensor out,const Tensor x,const Tensor xs,const Tensor w,const Tensor ws,size_t ro,size_t co)const override{
  this->Invocation(out,x,xs,w,ws,ro,co);if(!x.size(0))return;
  if(out.dtype()==DataType::kFloat32)launch<float>(out,x,xs,w,ws,ro,co);
  else if(out.dtype()==DataType::kBFloat16)launch<__nv_bfloat16>(out,x,xs,w,ws,ro,co);
  else launch<half>(out,x,xs,w,ws,ro,co);
  auto status=cudaGetLastError();if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
 }
private:
 template<class T>void launch(Tensor out,const Tensor x,const Tensor xs,const Tensor w,const Tensor ws,size_t ro,size_t co)const{
  fp8_block_detail::LinearBitsR17<T,Width><<<dim3(unsigned((w.size(0)+(256/Width)-1)/(256/Width)),unsigned(x.size(0))),256,0,reinterpret_cast<cudaStream_t>(this->stream_)>>>(reinterpret_cast<T*>(out.data()),reinterpret_cast<const uint8_t*>(x.data()),reinterpret_cast<const float*>(xs.data()),reinterpret_cast<const uint8_t*>(w.data()),reinterpret_cast<const float*>(ws.data()),x.size(0),w.size(0),x.size(1),ws.size(1),ro,co);
 }
};
}
