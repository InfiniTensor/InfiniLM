#pragma once
namespace infini::ops::mxfp4_detail {
// Keep the original256-thread product streams and firstthree high-bit add
// levels. Only the remaining five within32lane add levels use shuffles.
template<size_t N>__device__ __forceinline__ void hybridReduceR435(float(&v)[N],float*scratch){
 for(size_t i=0;i<N;++i)scratch[i*blockDim.x+threadIdx.x]=v[i];
 __syncthreads();
 for(size_t stride=128;stride>=32;stride>>=1){
  if(threadIdx.x<stride)for(size_t i=0;i<N;++i)scratch[i*blockDim.x+threadIdx.x]+=scratch[i*blockDim.x+threadIdx.x+stride];
  __syncthreads();
 }
 if(threadIdx.x<32){
  for(size_t i=0;i<N;++i)v[i]=scratch[i*blockDim.x+threadIdx.x];
  for(int shift=16;shift>0;shift>>=1)for(size_t i=0;i<N;++i){
#if defined(__HIPCC__)
   v[i]+=__shfl_down(v[i],shift,32);
#else
   v[i]+=__shfl_down_sync(0xffffffffu,v[i],shift,32);
#endif
  }
 }
 // Only thread0 consumes v after this helper; sharedstorage is never reused.
}
template <typename T>
__global__ void hybridW13R435(
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
    hybridReduceR435(sums, scratch);
    if (threadIdx.x == 0) {
        activated[route * intermediate_size + i]
            = mxfp4Store<T>(fusedMoeMxfp4Activate(sums[0], sums[1], activation));
    }
}


}
