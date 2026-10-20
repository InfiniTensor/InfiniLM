#ifndef __MXFP4_COMMON_CUDA_FUSED_MOE_MXFP4_KERNEL_CUH__
#define __MXFP4_COMMON_CUDA_FUSED_MOE_MXFP4_KERNEL_CUH__


#include "mxfp4_math.cuh"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace infini::ops::mxfp4_detail {

__device__ __forceinline__ float fusedMoeMxfp4Activate(
    float gate,
    float up,
    size_t activation) {
    if (activation == 2) {
        constexpr float beta = 4.0f;
        constexpr float linear_beta = 25.0f;
        const float situ_gate = beta * tanhf(gate / beta) / (1.0f + expf(-gate));
        const float bounded_up = linear_beta * tanhf(up / linear_beta);
        return situ_gate * bounded_up;
    }
    return gate / (1.0f + expf(-gate)) * up;
}

template <typename T>
__global__ void fusedMoeMxfp4W13Kernel(
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
        mxfp4DecodePair(
            gate_packed[packed_k], gate_scale[packed_k / 16], gate_low, gate_high);
        mxfp4DecodePair(
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
__global__ void fusedMoeMxfp4W2Kernel(
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
            mxfp4DecodePair(
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
__global__ void fusedMoeMxfp4W2Routes16Kernel(
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
                    mxfp4DecodePair(packed[row*(intermediate/2)+p],scales[row*(intermediate/32)+p/16],low,high);
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

template <typename T, typename Stream>
void launchFusedMoeMxfp4(
    T *output,
    T *activated,
    const T *input,
    const int32_t *selected_experts,
    const float *routing_weights,
    const uint8_t *w13_packed,
    const uint8_t *w13_scale,
    const uint8_t *w2_packed,
    const uint8_t *w2_scale,
    const Mxfp4MoeGeometry &info,
    Stream stream, bool parallel_routes = false) {
    constexpr size_t block_size = 256;
    // CUDA-compatible Hygon launches use a 32-bit total-thread range. Split
    // large prefill grids instead of letting grid_size * block_size overflow.
    constexpr size_t max_blocks_per_launch
        = std::numeric_limits<uint32_t>::max() / block_size;
    const size_t w13_grid = info.intermediate_size * info.routeCount();
    for (size_t block_offset = 0; block_offset < w13_grid;
         block_offset += max_blocks_per_launch) {
        const size_t grid_size = (w13_grid - block_offset < max_blocks_per_launch)
                                   ? w13_grid - block_offset
                                   : max_blocks_per_launch;
        fusedMoeMxfp4W13Kernel<<<grid_size, block_size,
                                 2 * block_size * sizeof(float), stream>>>(
            activated, input, selected_experts, w13_packed, w13_scale,
            info.routeCount(), info.topk, info.num_experts,
            info.hidden_size, info.intermediate_size, info.activation, block_offset);
    }

    const size_t w2_grid = info.hidden_size * info.num_tokens;
    for (size_t block_offset = 0; block_offset < w2_grid;
         block_offset += max_blocks_per_launch) {
        const size_t grid_size = (w2_grid - block_offset < max_blocks_per_launch)
                                   ? w2_grid - block_offset
                                   : max_blocks_per_launch;
        if(parallel_routes && info.topk<=16) {
            fusedMoeMxfp4W2Routes16Kernel<<<grid_size,block_size,0,stream>>>(
                output,activated,selected_experts,routing_weights,w2_packed,w2_scale,
                info.num_tokens,info.topk,info.num_experts,info.hidden_size,info.intermediate_size,block_offset);
        } else {
        fusedMoeMxfp4W2Kernel<<<grid_size, block_size,
                                block_size * sizeof(float), stream>>>(
            output, activated, selected_experts, routing_weights, w2_packed, w2_scale,
            info.num_tokens, info.topk, info.num_experts,
            info.hidden_size, info.intermediate_size, block_offset);
        }
    }
}

} // namespace infini::ops::mxfp4_detail

#endif
