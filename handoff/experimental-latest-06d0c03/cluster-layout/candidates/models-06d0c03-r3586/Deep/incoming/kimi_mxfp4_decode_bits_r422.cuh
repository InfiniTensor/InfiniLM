#pragma once
// Normal-range power-of-two scaling by exact FP32 bits. Extremes keep the
// original helper, including its signed-zero/subnormal/overflow behavior.
namespace infini::ops::mxfp4_detail {
__device__ __forceinline__ float decodeNormalR422(uint8_t code,uint8_t scale){
 unsigned magnitude=code&7,sign=unsigned(code&8)<<28;
 unsigned bits=sign;
 if(magnitude){unsigned exponent=unsigned(scale)-1+(magnitude>>1);unsigned mantissa=(magnitude>1&&(magnitude&1))?0x400000u:0u;bits|=(exponent<<23)|mantissa;}
 union{unsigned bits;float value;}v;v.bits=bits;return v.value;
}
__device__ __forceinline__ void decodePairR422(uint8_t packed,uint8_t scale,float&low,float&high){
 if(scale>=2&&scale<=252){low=decodeNormalR422(packed&15,scale);high=decodeNormalR422(packed>>4,scale);}
 else ::mxfp4DecodePair(packed,scale,low,high);
}
template <typename T>
__global__ void bitsW13R422(
    T *activated,
    const T *input,
    const int32_t *selected_experts,
    const uint8_t *w13_packed,
    const uint8_t *w13_scale,
    size_t route_count,
    size_t topk,
    size_t num_experts,
    size_t hidden_size,
    size_t intermediate_size,
    size_t activation,
    size_t block_offset) {
    const size_t block = block_offset + blockIdx.x;
    const size_t route = block / intermediate_size;
    const size_t i = block - route * intermediate_size;
    if (route >= route_count || i >= intermediate_size) {
        return;
    }

    const int32_t expert = selected_experts[route];
    if (expert < 0 || static_cast<size_t>(expert) >= num_experts) {
        if (threadIdx.x == 0) {
            activated[route * intermediate_size + i] = mxfp4Store<T>(0.0f);
        }
        return;
    }

    const size_t token = route / topk;
    const size_t packed_width = hidden_size / 2;
    const size_t scale_width = hidden_size / 32;
    const size_t gate_row = (static_cast<size_t>(expert) * 2 * intermediate_size + i);
    const size_t up_row = gate_row + intermediate_size;
    const auto *gate_packed = w13_packed + gate_row * packed_width;
    const auto *gate_scale = w13_scale + gate_row * scale_width;
    const auto *up_packed = w13_packed + up_row * packed_width;
    const auto *up_scale = w13_scale + up_row * scale_width;
    const auto *token_input = input + token * hidden_size;

    float sums[2] = {};
    for (size_t packed_k = threadIdx.x; packed_k < packed_width; packed_k += blockDim.x) {
        float gate_low;
        float gate_high;
        float up_low;
        float up_high;
        decodePairR422(
            gate_packed[packed_k], gate_scale[packed_k / 16], gate_low, gate_high);
        decodePairR422(
            up_packed[packed_k], up_scale[packed_k / 16], up_low, up_high);
        const size_t k = packed_k * 2;
        const float input_low = mxfp4Load(token_input, k);
        const float input_high = mxfp4Load(token_input, k + 1);
        sums[0] += input_low * gate_low + input_high * gate_high;
        sums[1] += input_low * up_low + input_high * up_high;
    }

    extern __shared__ float scratch[];
    mxfp4BlockReduce(sums, scratch);
    if (threadIdx.x == 0) {
        activated[route * intermediate_size + i]
            = mxfp4Store<T>(fusedMoeMxfp4Activate(sums[0], sums[1], activation));
    }
}

template <typename T>
__global__ void bitsW2R422(
    T* output,const T* activated,const int32_t* ids,const float* weights,
    const uint8_t* packed,const uint8_t* scales,size_t tokens,size_t unused_topk,
    size_t experts,size_t unused_hidden,size_t unused_intermediate,size_t offset) {
    constexpr size_t topk=16,hidden=3584,intermediate=384;
    const size_t index=offset+blockIdx.x, token=index/hidden, channel=index%hidden;
    if(token>=tokens)return;
    const size_t lane=threadIdx.x%32, group=threadIdx.x/32;
    __shared__ float sums[16];
    #pragma unroll 2
    for(size_t tile=0;tile<topk;tile+=8) {
        const size_t route_id=tile+group;
        float sum=0.0f;
        if(route_id<topk) {
            const int32_t expert=ids[token*topk+route_id];
            if(expert>=0 && static_cast<size_t>(expert)<experts) {
                const size_t row=static_cast<size_t>(expert)*hidden+channel;
                #pragma unroll 6
                for(size_t p=lane;p<intermediate/2;p+=32) {
                    float low,high;
                    decodePairR422(packed[row*(intermediate/2)+p],scales[row*(intermediate/32)+p/16],low,high);
                    sum+=mxfp4Load(activated,(token*topk+route_id)*intermediate+2*p)*low
                        +mxfp4Load(activated,(token*topk+route_id)*intermediate+2*p+1)*high;
                }
            }
        }
        for(int shift=16;shift>0;shift/=2) {
#if defined(__HIPCC__)
            sum+=__shfl_down(sum,shift,32);
#else
            sum+=__shfl_down_sync(0xffffffffu,sum,shift,32);
#endif
        }
        if(lane==0 && route_id<topk)sums[route_id]=sum;
    }
    __syncthreads();
    if(threadIdx.x==0) {
        float value=0.0f;
        for(size_t r=0;r<topk;++r) {
            const int32_t expert=ids[token*topk+r];
            if(expert>=0 && static_cast<size_t>(expert)<experts)
                value+=weights[token*topk+r]*sums[r];
        }
        output[index]=mxfp4Store<T>(value);
    }
}


}
