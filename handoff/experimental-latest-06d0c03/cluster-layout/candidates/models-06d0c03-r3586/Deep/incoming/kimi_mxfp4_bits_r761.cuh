#pragma once
__device__ __forceinline__ float mxfp4FastValueR760(unsigned code,unsigned scale){
 unsigned magnitude=code&7,sign=(code&8)<<12;if(!magnitude)return __uint_as_float(sign<<16);
 int exponent=int(scale)+int(magnitude/2)-1;unsigned mantissa=magnitude>=2&&(magnitude&1)?64:0,word;
 if(exponent>=255)word=0x7f80;
 else if(exponent<=0)word=(128+mantissa)>>(1-exponent);
 else word=(unsigned(exponent)<<7)|mantissa;
 return __uint_as_float((sign|word)<<16);
}
__device__ __forceinline__ void mxfp4FastDecodePairR760(uint8_t packed,uint8_t scale,float&lo,float&hi){lo=mxfp4FastValueR760(packed&15,scale);hi=mxfp4FastValueR760(packed>>4,scale);}
