// SPDX-License-Identifier: Apache-2.0
// Adapted from vLLM at commit ffc4f08c8ee130d4ea6347c1bf31ffd4f8af28ab:
// csrc/libtorch_stable/moe/moe_wna16.cu
// csrc/libtorch_stable/moe/moe_wna16_utils.h

#ifndef INFINI_OPS_NVIDIA_MOE_WNA16_GEMM_KERNEL_CUH_
#define INFINI_OPS_NVIDIA_MOE_WNA16_GEMM_KERNEL_CUH_

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <type_traits>

namespace infini::ops::moe_wna16_gemm_detail {

template <typename Data>
struct ScalarType;

template <>
struct ScalarType<half> {
  using Data2 = half2;

  static __device__ float ToFloat(half value) { return __half2float(value); }

  static __device__ half2 Broadcast(half value) { return __half2half2(value); }

  static __device__ half FromFloat(float value) { return __float2half(value); }

  static __device__ half FromInt(int value) { return __int2half_rn(value); }
};

template <>
struct ScalarType<__nv_bfloat16> {
  using Data2 = __nv_bfloat162;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  static __device__ float ToFloat(__nv_bfloat16 value) {
    return __bfloat162float(value);
  }

  static __device__ __nv_bfloat162 Broadcast(__nv_bfloat16 value) {
    return __bfloat162bfloat162(value);
  }

  static __device__ __nv_bfloat16 FromFloat(float value) {
    return __float2bfloat16(value);
  }

  static __device__ __nv_bfloat16 FromInt(int value) {
    return __int2bfloat16_rn(value);
  }
#endif
};

template <int kLut>
__device__ inline int Lop3(int a, int b, int c) {
  int result;
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(result)
               : "r"(a), "r"(b), "r"(c), "n"(kLut));
  return result;
}

template <int kStartByte, int kMask>
__device__ inline uint32_t Prmt(uint32_t value) {
  uint32_t result;
  asm volatile("prmt.b32 %0, %1, %2, %3;\n"
               : "=r"(result)
               : "r"(value), "n"(kStartByte), "n"(kMask));
  return result;
}

template <typename Data2, int kBit>
__device__ inline void Dequantize(uint32_t value, Data2* output);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
template <>
__device__ inline void Dequantize<half2, 4>(uint32_t value, half2* output) {
  constexpr int kLowMask = 0x000f000f;
  constexpr int kHighMask = 0x00f000f0;
  constexpr int kExponent = 0x64006400;
  constexpr int kSubtract = 0x64006400;
  constexpr int kMultiply = 0x2c002c00;
  constexpr int kAdd = 0xd400d400;

  int low0 = Lop3<(0xf0 & 0xcc) | 0xaa>(value, kLowMask, kExponent);
  int high0 = Lop3<(0xf0 & 0xcc) | 0xaa>(value, kHighMask, kExponent);
  value >>= 8;
  int low1 = Lop3<(0xf0 & 0xcc) | 0xaa>(value, kLowMask, kExponent);
  int high1 = Lop3<(0xf0 & 0xcc) | 0xaa>(value, kHighMask, kExponent);

  output[0] = __hsub2(*reinterpret_cast<half2*>(&low0),
                      *reinterpret_cast<const half2*>(&kSubtract));
  output[1] = __hfma2(*reinterpret_cast<half2*>(&high0),
                      *reinterpret_cast<const half2*>(&kMultiply),
                      *reinterpret_cast<const half2*>(&kAdd));
  output[2] = __hsub2(*reinterpret_cast<half2*>(&low1),
                      *reinterpret_cast<const half2*>(&kSubtract));
  output[3] = __hfma2(*reinterpret_cast<half2*>(&high1),
                      *reinterpret_cast<const half2*>(&kMultiply),
                      *reinterpret_cast<const half2*>(&kAdd));
}

template <>
__device__ inline void Dequantize<half2, 8>(uint32_t value, half2* output) {
  constexpr uint32_t kLowMask = 0x5250;
  constexpr uint32_t kHighMask = 0x5351;
  constexpr uint32_t kStartByte = 0x64646464;
  constexpr uint32_t kMagic = 0x64006400;

  uint32_t low = Prmt<kStartByte, kLowMask>(value);
  uint32_t high = Prmt<kStartByte, kHighMask>(value);
  output[0] = __hsub2(*reinterpret_cast<half2*>(&low),
                      *reinterpret_cast<const half2*>(&kMagic));
  output[1] = __hsub2(*reinterpret_cast<half2*>(&high),
                      *reinterpret_cast<const half2*>(&kMagic));
}
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
template <>
__device__ inline void Dequantize<__nv_bfloat162, 4>(uint32_t value,
                                                     __nv_bfloat162* output) {
  constexpr int kMask = 0x000f000f;
  constexpr int kExponent = 0x43004300;
  constexpr uint32_t kMultiply = 0x3f803f80;
  constexpr uint32_t kAdd = 0xc300c300;

  int low0 = Lop3<(0xf0 & 0xcc) | 0xaa>(value, kMask, kExponent);
  value >>= 4;
  int high0 = Lop3<(0xf0 & 0xcc) | 0xaa>(value, kMask, kExponent);
  value >>= 4;
  int low1 = Lop3<(0xf0 & 0xcc) | 0xaa>(value, kMask, kExponent);
  value >>= 4;
  int high1 = Lop3<(0xf0 & 0xcc) | 0xaa>(value, kMask, kExponent);

  output[0] = __hfma2(*reinterpret_cast<__nv_bfloat162*>(&low0),
                      *reinterpret_cast<const __nv_bfloat162*>(&kMultiply),
                      *reinterpret_cast<const __nv_bfloat162*>(&kAdd));
  output[1] = __hfma2(*reinterpret_cast<__nv_bfloat162*>(&high0),
                      *reinterpret_cast<const __nv_bfloat162*>(&kMultiply),
                      *reinterpret_cast<const __nv_bfloat162*>(&kAdd));
  output[2] = __hfma2(*reinterpret_cast<__nv_bfloat162*>(&low1),
                      *reinterpret_cast<const __nv_bfloat162*>(&kMultiply),
                      *reinterpret_cast<const __nv_bfloat162*>(&kAdd));
  output[3] = __hfma2(*reinterpret_cast<__nv_bfloat162*>(&high1),
                      *reinterpret_cast<const __nv_bfloat162*>(&kMultiply),
                      *reinterpret_cast<const __nv_bfloat162*>(&kAdd));
}

template <>
__device__ inline void Dequantize<__nv_bfloat162, 8>(uint32_t value,
                                                     __nv_bfloat162* output) {
  float intermediates[4];
  auto* bits = reinterpret_cast<uint32_t*>(intermediates);
  constexpr uint32_t kBase = 0x4b000000;

  bits[0] = __byte_perm(value, kBase, 0x7650);
  bits[1] = __byte_perm(value, kBase, 0x7652);
  bits[2] = __byte_perm(value, kBase, 0x7651);
  bits[3] = __byte_perm(value, kBase, 0x7653);
  for (int i = 0; i < 4; ++i) {
    intermediates[i] -= 8388608.0f;
  }

  auto* output_bits = reinterpret_cast<uint32_t*>(output);
  output_bits[0] = __byte_perm(bits[0], bits[1], 0x7632);
  output_bits[1] = __byte_perm(bits[2], bits[3], 0x7632);
}
#endif

template <typename Data, int kBit, int kGroups>
__global__ void MoeWna16GemmKernel(
    const Data* __restrict__ input, Data* __restrict__ output,
    const uint32_t* __restrict__ qweight, const Data* __restrict__ scales,
    const uint32_t* __restrict__ qzeros, const float* __restrict__ topk_weights,
    const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_post_pad,
    uint64_t sorted_token_ids_size, uint16_t group_size, uint16_t top_k,
    uint32_t size_m, uint32_t size_n, uint32_t size_k, uint16_t block_size_m,
    uint16_t block_size_n, uint16_t block_size_k, bool has_zero_point,
    bool multiply_topk_weight) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 700
  // The common CUDA build includes lower virtual architectures; runtime rejects
  // devices below SM70 before this implementation is selected.
  return;
#else
#if __CUDA_ARCH__ < 800
  if constexpr (std::is_same_v<Data, __nv_bfloat16>) {
    return;
  } else {
#endif
    using Traits = ScalarType<Data>;
    using Data2 = typename Traits::Data2;

    if (blockIdx.x * block_size_m >= num_tokens_post_pad[0]) {
      return;
    }

    const int32_t offset_n = blockIdx.y * block_size_n + threadIdx.x;
    const int32_t offset_k = blockIdx.z * block_size_k;
    const int32_t expert_id = expert_ids[blockIdx.x];

    int32_t num_valid_tokens = 0;
    extern __shared__ uint16_t shared_storage[];
    auto* block_input = reinterpret_cast<Data*>(shared_storage);
    auto* block_input2 = reinterpret_cast<Data2*>(block_input);

    for (int m = 0; m < block_size_m; ++m) {
      const uint64_t route =
          static_cast<uint64_t>(blockIdx.x) * block_size_m + m;
      if (route >= sorted_token_ids_size) {
        break;
      }
      const int32_t token_index = sorted_token_ids[route];
      if (token_index / top_k >= size_m) {
        break;
      }

      num_valid_tokens = m + 1;
      if (expert_id == -1) {
        continue;
      }

      const int k_per_thread = (block_size_k + block_size_n - 1) / block_size_n;
      for (int i = 0; i < k_per_thread; ++i) {
        const int k = block_size_n * i + threadIdx.x;
        if (k >= block_size_k || offset_k + k >= size_k) {
          break;
        }

        int original_k;
        if constexpr (kBit == 4) {
          const int order = (threadIdx.x % 2) * 4 + (threadIdx.x % 8) / 2;
          original_k = block_size_n * i + threadIdx.x / 8 * 8 + order;
        } else {
          const int order = (threadIdx.x % 2) * 2 + (threadIdx.x % 4) / 2;
          original_k = block_size_n * i + threadIdx.x / 4 * 4 + order;
        }

        const int64_t input_offset =
            static_cast<int64_t>(token_index / top_k) * size_k + offset_k +
            original_k;
        block_input[m * block_size_k + k] = input[input_offset];
      }
    }

    if (expert_id == -1) {
      return;
    }
    __syncthreads();
    if (threadIdx.x >= block_size_n || offset_n >= size_n) {
      return;
    }

    float result[64];
    Data2 result2;
    Data2 scale2;
    Data2 zero_point2;

    constexpr int kValuesPerWord = 32 / kBit;
    const uint64_t expert_offset =
        static_cast<uint64_t>(size_n) * size_k * expert_id;
    const uint32_t* expert_qweight = qweight + expert_offset / kValuesPerWord;
    const Data* expert_scales = scales + expert_offset / group_size;
    const uint32_t* expert_qzeros =
        has_zero_point ? qzeros + expert_offset / group_size / kValuesPerWord
                       : nullptr;

    alignas(16) uint32_t packed_weight[4];
    auto* packed_weight4 = reinterpret_cast<float4*>(packed_weight);

    alignas(16) Data group_scales[kGroups];
    const int scales_offset =
        (offset_n * size_k + offset_k) / group_size / kGroups;
    if constexpr (kGroups == 1) {
      group_scales[0] = expert_scales[scales_offset];
    } else if constexpr (kGroups == 2) {
      *reinterpret_cast<float*>(group_scales) =
          reinterpret_cast<const float*>(expert_scales)[scales_offset];
    } else if constexpr (kGroups == 4) {
      *reinterpret_cast<float2*>(group_scales) =
          reinterpret_cast<const float2*>(expert_scales)[scales_offset];
    } else {
      *reinterpret_cast<float4*>(group_scales) =
          reinterpret_cast<const float4*>(expert_scales)[scales_offset];
    }

    alignas(8) uint8_t group_zero_points[kGroups];
    if (!has_zero_point) {
      zero_point2 = Traits::Broadcast(Traits::FromInt(kBit == 4 ? 8 : 128));
    } else {
      const int zero_point_offset =
          (offset_n / (8 / kBit)) * (size_k / group_size / kGroups) +
          offset_k / group_size / kGroups;
      if constexpr (kGroups == 1) {
        group_zero_points[0] =
            reinterpret_cast<const uint8_t*>(expert_qzeros)[zero_point_offset];
      } else if constexpr (kGroups == 2) {
        *reinterpret_cast<uint16_t*>(group_zero_points) =
            reinterpret_cast<const uint16_t*>(expert_qzeros)[zero_point_offset];
      } else if constexpr (kGroups == 4) {
        *reinterpret_cast<uint32_t*>(group_zero_points) =
            reinterpret_cast<const uint32_t*>(expert_qzeros)[zero_point_offset];
      } else {
        *reinterpret_cast<uint64_t*>(group_zero_points) =
            reinterpret_cast<const uint64_t*>(expert_qzeros)[zero_point_offset];
      }
    }

    for (int packed_k = 0; packed_k < block_size_k / kValuesPerWord;
         ++packed_k) {
      const int k = offset_k + packed_k * kValuesPerWord;
      if (k >= size_k) {
        break;
      }
      const int32_t weight_offset = offset_n * size_k + k;

      if (packed_k % 4 == 0) {
        *packed_weight4 = reinterpret_cast<const float4*>(
            expert_qweight)[weight_offset / kValuesPerWord / 4];
      }

      const int packed_values_per_group = group_size / kValuesPerWord;
      if (packed_k % packed_values_per_group == 0) {
        const int group = packed_k / packed_values_per_group;
        scale2 = Traits::Broadcast(group_scales[group]);

        if (has_zero_point) {
          uint8_t zero_point = group_zero_points[group];
          if constexpr (kBit == 4) {
            zero_point = (zero_point >> ((threadIdx.x % 2) * 4)) & 0x0f;
          }
          zero_point2 = Traits::Broadcast(Traits::FromInt(zero_point));
        }
      }

      Data2 dequantized[16 / kBit];
      Dequantize<Data2, kBit>(packed_weight[packed_k % 4], dequantized);

      for (int m = 0; m < num_valid_tokens; ++m) {
        result2 = {};
#pragma unroll
        for (int i = 0; i < 16 / kBit; ++i) {
          const int32_t input_offset =
              m * block_size_k / 2 + packed_k * (16 / kBit) + i;
          result2 =
              __hfma2(__hmul2(__hsub2(dequantized[i], zero_point2), scale2),
                      block_input2[input_offset], result2);
        }

        const float partial =
            Traits::ToFloat(result2.x) + Traits::ToFloat(result2.y);
        if (packed_k == 0) {
          result[m] = partial;
        } else {
          result[m] += partial;
        }
      }
    }

    for (int m = 0; m < num_valid_tokens; ++m) {
      const int32_t token_index =
          sorted_token_ids[blockIdx.x * block_size_m + m];
      if (multiply_topk_weight) {
        result[m] *= topk_weights[token_index];
      }
      const int64_t output_offset =
          static_cast<int64_t>(token_index) * size_n + offset_n;
      atomicAdd(output + output_offset, Traits::FromFloat(result[m]));
    }
#if __CUDA_ARCH__ < 800
  }
#endif
#endif
}

}  // namespace infini::ops::moe_wna16_gemm_detail

#endif  // INFINI_OPS_NVIDIA_MOE_WNA16_GEMM_KERNEL_CUH_
