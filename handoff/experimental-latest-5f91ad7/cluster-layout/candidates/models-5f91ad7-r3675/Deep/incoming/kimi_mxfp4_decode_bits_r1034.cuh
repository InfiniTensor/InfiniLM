#pragma once
namespace infini::ops::mxfp4_detail {
// E2M1 values are signed {0,.5,1,1.5,2,3,4,6}. For E8M0 codes2..252,
// every nonzero result is normal FP32 and its significand is exact.
// Preserve the original ldexp path at underflow/overflow boundaries.
__device__ __forceinline__ float nibbleBitsR1034(unsigned code,unsigned scale){
 unsigned magnitude=code&7,sign=(code&8u)<<28;
 if(!magnitude)return __uint_as_float(sign);
 int exponent=int(scale)+(magnitude==1?-1:int(magnitude>>1)-1);
 unsigned fraction=(magnitude>=3&&(magnitude&1))?0x400000u:0u;
 return __uint_as_float(sign|(unsigned(exponent)<<23)|fraction);
}
__device__ __forceinline__ void decodePairBitsR1034(uint8_t packed,uint8_t scale,float&low,float&high){
 if(scale<2||scale>252){mxfp4DecodePair(packed,scale,low,high);return;}
 low=nibbleBitsR1034(packed&15u,scale);high=nibbleBitsR1034(packed>>4,scale);
}
__global__ void decodeProofR1034(uint32_t*reference,uint32_t*candidate){
 unsigned index=blockIdx.x*blockDim.x+threadIdx.x;if(index>=65536)return;
 uint8_t packed=uint8_t(index),scale=uint8_t(index>>8);float lo,hi;
 mxfp4DecodePair(packed,scale,lo,hi);reference[index*2]=__float_as_uint(lo);reference[index*2+1]=__float_as_uint(hi);
 decodePairBitsR1034(packed,scale,lo,hi);candidate[index*2]=__float_as_uint(lo);candidate[index*2+1]=__float_as_uint(hi);
}
}
