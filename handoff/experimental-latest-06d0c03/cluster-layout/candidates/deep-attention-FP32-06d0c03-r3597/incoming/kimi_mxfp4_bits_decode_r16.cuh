#pragma once
// Experiment: construct normal FP32 E2M1*E8M0 weights directly. Original
// ldexpf path handles all edge scales. No accumulator/reduction-tree changes.
__device__ __forceinline__ float mxfp4DecodeBitsNormal(unsigned code,unsigned scale){
 unsigned magnitude=code&7u;
 unsigned bits=magnitude?((scale-1u+(magnitude>>1))<<23)|((magnitude>1u?(magnitude&1u):0u)<<22):0u;
 return __uint_as_float(bits|((code&8u)<<28));
}
__device__ __forceinline__ void mxfp4DecodePairBits(uint8_t packed,uint8_t scale,float&low,float&high){
 if(scale>=2&&scale<=252){low=mxfp4DecodeBitsNormal(packed&15u,scale);high=mxfp4DecodeBitsNormal(packed>>4,scale);}
 else mxfp4DecodePair(packed,scale,low,high);
}
namespace infini::ops::mxfp4_detail {
template <typename T>
__global__ void W13BitsKernel(
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
        mxfp4DecodePairBits(
            gate_packed[packed_k], gate_scale[packed_k / 16], gate_low, gate_high);
        mxfp4DecodePairBits(
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
__global__ void W2UnusedBitsKernel(
    T *output,
    const T *activated,
    const int32_t *selected_experts,
    const float *routing_weights,
    const uint8_t *w2_packed,
    const uint8_t *w2_scale,
    size_t num_tokens,
    size_t topk,
    size_t num_experts,
    size_t hidden_size,
    size_t intermediate_size,
    size_t block_offset) {
    const size_t block = block_offset + blockIdx.x;
    const size_t token = block / hidden_size;
    const size_t h = block - token * hidden_size;
    if (token >= num_tokens || h >= hidden_size) {
        return;
    }

    const size_t packed_width = intermediate_size / 2;
    const size_t scale_width = intermediate_size / 32;
    float output_value = 0.0f;
    extern __shared__ float scratch[];
    for (size_t route_index = 0; route_index < topk; ++route_index) {
        const size_t route = token * topk + route_index;
        const int32_t expert = selected_experts[route];
        if (expert < 0 || static_cast<size_t>(expert) >= num_experts) {
            continue;
        }
        const size_t weight_row = static_cast<size_t>(expert) * hidden_size + h;
        const auto *packed_row = w2_packed + weight_row * packed_width;
        const auto *scale_row = w2_scale + weight_row * scale_width;
        const auto *route_input = activated + route * intermediate_size;

        float sum[1] = {};
        for (size_t packed_k = threadIdx.x; packed_k < packed_width; packed_k += blockDim.x) {
            float weight_low;
            float weight_high;
            mxfp4DecodePairBits(
                packed_row[packed_k], scale_row[packed_k / 16], weight_low, weight_high);
            const size_t k = packed_k * 2;
            sum[0] += mxfp4Load(route_input, k) * weight_low
                    + mxfp4Load(route_input, k + 1) * weight_high;
        }
        mxfp4BlockReduce(sum, scratch);
        if (threadIdx.x == 0) {
            output_value += routing_weights[route] * sum[0];
        }
    }
    if (threadIdx.x == 0) {
        output[token * hidden_size + h] = mxfp4Store<T>(output_value);
    }
}

template <typename T>
__global__ void W2RoutesBitsKernel(
    T* output,const T* activated,const int32_t* ids,const float* weights,
    const uint8_t* packed,const uint8_t* scales,size_t tokens,size_t topk,
    size_t experts,size_t hidden,size_t intermediate,size_t offset) {
    const size_t index=offset+blockIdx.x, token=index/hidden, channel=index%hidden;
    if(token>=tokens)return;
    const size_t lane=threadIdx.x%32, group=threadIdx.x/32;
    __shared__ float sums[16];
    for(size_t tile=0;tile<topk;tile+=8) {
        const size_t route_id=tile+group;
        float sum=0.0f;
        if(route_id<topk) {
            const int32_t expert=ids[token*topk+route_id];
            if(expert>=0 && static_cast<size_t>(expert)<experts) {
                const size_t row=static_cast<size_t>(expert)*hidden+channel;
                for(size_t p=lane;p<intermediate/2;p+=32) {
                    float low,high;
                    mxfp4DecodePairBits(packed[row*(intermediate/2)+p],scales[row*(intermediate/32)+p/16],low,high);
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


template<class T>void LaunchBitsDecode(T*out,T*scratch,const T*input,
 const int32_t*ids,const float*weights,const uint8_t*w13,const uint8_t*s13,
 const uint8_t*w2,const uint8_t*s2,size_t tokens,size_t experts,size_t hidden,
 size_t intermediate,size_t topk,size_t activation,int mode,cudaStream_t stream){
 size_t work=tokens*topk*intermediate;
 if(mode==2||mode==4)W13BitsKernel<<<unsigned(work),256,2048,stream>>>(scratch,input,ids,w13,s13,tokens*topk,topk,experts,hidden,intermediate,activation,size_t{0});
 else fusedMoeMxfp4W13Kernel<<<unsigned(work),256,2048,stream>>>(scratch,input,ids,w13,s13,tokens*topk,topk,experts,hidden,intermediate,activation,size_t{0});
 if(mode==2||mode==3)W2RoutesBitsKernel<<<unsigned(tokens*hidden),256,0,stream>>>(out,scratch,ids,weights,w2,s2,tokens,topk,experts,hidden,intermediate,size_t{0});
 else fusedMoeMxfp4W2Routes16Kernel<<<unsigned(tokens*hidden),256,0,stream>>>(out,scratch,ids,weights,w2,s2,tokens,topk,experts,hidden,intermediate,size_t{0});
}
}
__global__ void KimiBitsDecodeDomain(float*out){
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=65536)return;
 float a,b,c,d;mxfp4DecodePair(i&255u,i>>8,a,b);mxfp4DecodePairBits(i&255u,i>>8,c,d);
 out[4*i]=a;out[4*i+1]=b;out[4*i+2]=c;out[4*i+3]=d;
}
extern "C" int modern_moe_bits_domain(void*out,void*stream){KimiBitsDecodeDomain<<<256,256,0,static_cast<cudaStream_t>(stream)>>>(static_cast<float*>(out));return int(cudaGetLastError());}
