#ifndef INFINI_OPS_CUDA_RANDOM_SAMPLE_INFINILM_KERNEL_CUH_
#define INFINI_OPS_CUDA_RANDOM_SAMPLE_INFINILM_KERNEL_CUH_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>
#include <type_traits>

#include "data_type.h"
#include "native/cuda/caster.cuh"
#include "native/cuda/runtime_.h"

namespace infini::ops {

namespace random_sample_infinilm_detail {

constexpr std::size_t Align256(std::size_t size) {
  return (size + 255) & ~std::size_t{255};
}

template <Device::Type kDev, typename T>
inline constexpr bool kUseFloatCubForMooreBFloat16 =
    kDev == Device::Type::kMoore && IsBFloat16<kDev, T>;

template <typename TIn, typename TOut>
__global__ void ConvertKernel(const TIn* __restrict__ in,
                              TOut* __restrict__ out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = static_cast<TOut>(in[i]);
  }
}

template <typename TDst, typename TSrc>
__global__ void ConvertKeyValuePairKernel(
    cub::KeyValuePair<int, TDst>* __restrict__ dst,
    const cub::KeyValuePair<int, TSrc>* __restrict__ src) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    dst->key = src->key;
    dst->value = static_cast<TDst>(src->value);
  }
}

template <Device::Type kDev, typename T, typename Stream>
auto ArgMaxCub(cub::KeyValuePair<int, T>* kv_pair, const T* logits, int n,
               void* workspace_ptr, std::size_t& workspace_len, Stream stream) {
  if constexpr (kUseFloatCubForMooreBFloat16<kDev, T>) {
    auto* cursor = static_cast<char*>(workspace_ptr);

    auto* temp_kv = reinterpret_cast<cub::KeyValuePair<int, float>*>(cursor);
    cursor += Align256(sizeof(cub::KeyValuePair<int, float>));
    workspace_len -= Align256(sizeof(cub::KeyValuePair<int, float>));

    auto* temp_logits = reinterpret_cast<float*>(cursor);
    cursor += Align256(sizeof(float) * static_cast<std::size_t>(n));
    workspace_len -= Align256(sizeof(float) * static_cast<std::size_t>(n));

    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    ConvertKernel<<<grid_size, block_size, 0, stream>>>(logits, temp_logits, n);

    auto err = cub::DeviceReduce::ArgMax(cursor, workspace_len, temp_logits,
                                         temp_kv, n, stream);
    if (err != 0) {
      return err;
    }

    ConvertKeyValuePairKernel<<<1, 1, 0, stream>>>(kv_pair, temp_kv);

    return err;
  } else {
    return cub::DeviceReduce::ArgMax(workspace_ptr, workspace_len, logits,
                                     kv_pair, n, stream);
  }
}

template <Device::Type kDev, typename Tval, typename Tidx, typename Stream>
auto RadixSortCub(void* workspace_ptr, std::size_t& workspace_len,
                  const Tval* key_in, Tval* key_out, const Tidx* val_in,
                  Tidx* val_out, int n, Stream stream) {
  if constexpr (kUseFloatCubForMooreBFloat16<kDev, Tval>) {
    auto* cursor = static_cast<char*>(workspace_ptr);

    auto* temp_key_in = reinterpret_cast<float*>(cursor);
    cursor += Align256(sizeof(float) * static_cast<std::size_t>(n));
    workspace_len -= Align256(sizeof(float) * static_cast<std::size_t>(n));

    auto* temp_key_out = reinterpret_cast<float*>(cursor);
    cursor += Align256(sizeof(float) * static_cast<std::size_t>(n));
    workspace_len -= Align256(sizeof(float) * static_cast<std::size_t>(n));

    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    ConvertKernel<<<grid_size, block_size, 0, stream>>>(key_in, temp_key_in, n);

    auto err = cub::DeviceRadixSort::SortPairsDescending(
        cursor, workspace_len, temp_key_in, temp_key_out, val_in, val_out, n, 0,
        sizeof(float) * 8, stream);
    if (err != 0) {
      return err;
    }

    ConvertKernel<<<grid_size, block_size, 0, stream>>>(temp_key_out, key_out,
                                                        n);

    return err;
  } else {
    return cub::DeviceRadixSort::SortPairsDescending(
        workspace_ptr, workspace_len, key_in, key_out, val_in, val_out, n, 0,
        sizeof(Tval) * 8, stream);
  }
}

template <Device::Type kDev, typename T, typename Stream>
auto InclusiveSumCub(void* workspace_ptr, std::size_t& workspace_len, T* data,
                     int n, Stream stream) {
  if constexpr (kUseFloatCubForMooreBFloat16<kDev, T>) {
    auto* cursor = static_cast<char*>(workspace_ptr);

    auto* temp_data = reinterpret_cast<float*>(cursor);
    cursor += Align256(sizeof(float) * static_cast<std::size_t>(n));
    workspace_len -= Align256(sizeof(float) * static_cast<std::size_t>(n));

    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    ConvertKernel<<<grid_size, block_size, 0, stream>>>(data, temp_data, n);

    auto err = cub::DeviceScan::InclusiveSum(cursor, workspace_len, temp_data,
                                             temp_data, n, stream);
    if (err != 0) {
      return err;
    }

    ConvertKernel<<<grid_size, block_size, 0, stream>>>(temp_data, data, n);

    return err;
  } else {
    return cub::DeviceScan::InclusiveSum(workspace_ptr, workspace_len, data,
                                         data, n, stream);
  }
}

template <Device::Type kDev, typename Tidx, typename Tval>
std::size_t WorkspaceSize(std::size_t n_) {
  int n = static_cast<int>(n_);

  std::size_t argmax_workspace = 0;
  std::size_t sort_workspace = 0;
  std::size_t scan_workspace = 0;

  if constexpr (kUseFloatCubForMooreBFloat16<kDev, Tval>) {
    cub::DeviceReduce::ArgMax(
        nullptr, argmax_workspace, static_cast<const float*>(nullptr),
        static_cast<cub::KeyValuePair<int, float>*>(nullptr), n, nullptr);
    cub::DeviceRadixSort::SortPairsDescending(
        nullptr, sort_workspace, static_cast<const float*>(nullptr),
        static_cast<float*>(nullptr), static_cast<const Tidx*>(nullptr),
        static_cast<Tidx*>(nullptr), n, 0, sizeof(float) * 8, nullptr);
    cub::DeviceScan::InclusiveSum(nullptr, scan_workspace,
                                  static_cast<float*>(nullptr),
                                  static_cast<float*>(nullptr), n, nullptr);

    argmax_workspace += 256;
    argmax_workspace += Align256(sizeof(cub::KeyValuePair<int, float>));
    argmax_workspace += Align256(sizeof(float) * n_);

    std::size_t random_workspace = Align256(sizeof(Tidx) * n_);
    random_workspace += Align256(sizeof(Tval) * n_);
    random_workspace += Align256(sizeof(Tidx) * n_);
    random_workspace +=
        std::max(Align256(sizeof(float) * n_) * 2 + sort_workspace,
                 Align256(sizeof(float) * n_) + scan_workspace);

    return std::max(argmax_workspace, random_workspace);
  } else {
    cub::DeviceReduce::ArgMax(
        nullptr, argmax_workspace, static_cast<const Tval*>(nullptr),
        static_cast<cub::KeyValuePair<int, Tval>*>(nullptr), n, nullptr);
    cub::DeviceRadixSort::SortPairsDescending(
        nullptr, sort_workspace, static_cast<const Tval*>(nullptr),
        static_cast<Tval*>(nullptr), static_cast<const Tidx*>(nullptr),
        static_cast<Tidx*>(nullptr), n, 0, sizeof(Tval) * 8, nullptr);
    cub::DeviceScan::InclusiveSum(nullptr, scan_workspace,
                                  static_cast<Tval*>(nullptr),
                                  static_cast<Tval*>(nullptr), n, nullptr);

    argmax_workspace += 256;

    std::size_t random_workspace = Align256(sizeof(Tidx) * n_);
    random_workspace += Align256(sizeof(Tval) * n_);
    random_workspace += Align256(sizeof(Tidx) * n_);
    random_workspace += std::max(sort_workspace, scan_workspace);

    return std::max(argmax_workspace, random_workspace);
  }
}

template <typename Tidx, typename Tval>
__global__ void CastIndexKernel(Tidx* __restrict__ result,
                                const cub::KeyValuePair<int, Tval>* kv_pair) {
  *result = static_cast<Tidx>(kv_pair->key);
}

template <typename Tidx>
__global__ void FillIndicesKernel(Tidx* __restrict__ indices, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    indices[i] = static_cast<Tidx>(i);
  }
}

template <Device::Type kDev, typename T>
__global__ void PartialSoftmaxKernel(T* __restrict__ data, int n,
                                     float temperature) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (0 < i && i < n) {
    float max_value = Caster<kDev>::template Cast<float>(data[0]);
    float value = Caster<kDev>::template Cast<float>(data[i]);
    data[i] =
        Caster<kDev>::template Cast<T>(expf((value - max_value) / temperature));
  }
}

template <Device::Type kDev, typename T>
__global__ void SetSoftmaxMaxKernel(T* __restrict__ data) {
  *data = Caster<kDev>::template Cast<T>(1.0f);
}

template <Device::Type kDev, typename Tval, typename Tidx>
__global__ void RandomSampleInfinilmKernel(Tidx* __restrict__ result,
                                           const Tval* __restrict__ sorted,
                                           const Tidx* __restrict__ indices_out,
                                           std::size_t n, float random,
                                           float topp, std::size_t topk) {
  topk = topk < n ? topk : n;
  float total = Caster<kDev>::template Cast<float>(sorted[n - 1]);
  float topk_sum = Caster<kDev>::template Cast<float>(sorted[topk - 1]);
  float threshold = random * fminf(topp * total, topk_sum);

  for (std::size_t i = 0;; ++i) {
    if (Caster<kDev>::template Cast<float>(sorted[i]) >= threshold) {
      *result = indices_out[i];
      return;
    }
  }
}

}  // namespace random_sample_infinilm_detail

template <int block_size, Device::Type kDev, typename Tidx, typename Tval>
void RandomSampleInfinilmArgmax(void* workspace, std::size_t workspace_size,
                                Tidx* result, const Tval* logits, std::size_t n,
                                void* stream_) {
  auto stream = reinterpret_cast<typename Runtime<kDev>::Stream>(stream_);
  auto* kv_pair = reinterpret_cast<cub::KeyValuePair<int, Tval>*>(workspace);
  auto* cub_workspace = static_cast<char*>(workspace) + 256;
  workspace_size -= 256;

  auto err = random_sample_infinilm_detail::ArgMaxCub<kDev>(
      kv_pair, logits, static_cast<int>(n), cub_workspace, workspace_size,
      stream);
  assert(err == 0 && "`RandomSampleInfinilm` CUB ArgMax failed");

  random_sample_infinilm_detail::CastIndexKernel<<<1, 1, 0, stream>>>(result,
                                                                      kv_pair);
}

template <int block_size, Device::Type kDev, typename Tidx, typename Tval>
void RandomSampleInfinilmTopP(void* workspace, std::size_t workspace_size,
                              Tidx* result, const Tval* logits, std::size_t n,
                              float random_val, float topp, int topk,
                              float temperature, void* stream_) {
  auto stream = reinterpret_cast<typename Runtime<kDev>::Stream>(stream_);
  auto workspace_begin = reinterpret_cast<std::uintptr_t>(workspace);
  auto workspace_end = workspace_begin + workspace_size;
  auto workspace_cursor = workspace_begin;

  auto* indices = reinterpret_cast<Tidx*>(workspace_cursor);
  workspace_cursor += random_sample_infinilm_detail::Align256(sizeof(Tidx) * n);

  auto* sorted = reinterpret_cast<Tval*>(workspace_cursor);
  workspace_cursor += random_sample_infinilm_detail::Align256(sizeof(Tval) * n);

  auto* indices_out = reinterpret_cast<Tidx*>(workspace_cursor);
  workspace_cursor += random_sample_infinilm_detail::Align256(sizeof(Tidx) * n);

  auto* cub_workspace = reinterpret_cast<void*>(workspace_cursor);
  workspace_size = workspace_end - workspace_cursor;

  auto block = static_cast<unsigned>(std::min<std::size_t>(block_size, n));
  auto grid = static_cast<unsigned>((n + block - 1) / block);

  random_sample_infinilm_detail::FillIndicesKernel<<<grid, block, 0, stream>>>(
      indices, static_cast<int>(n));

  auto err = random_sample_infinilm_detail::RadixSortCub<kDev>(
      cub_workspace, workspace_size, logits, sorted, indices, indices_out,
      static_cast<int>(n), stream);
  assert(err == 0 && "`RandomSampleInfinilm` CUB radix sort failed");

  random_sample_infinilm_detail::PartialSoftmaxKernel<kDev>
      <<<grid, block, 0, stream>>>(sorted, static_cast<int>(n), temperature);
  random_sample_infinilm_detail::SetSoftmaxMaxKernel<kDev>
      <<<1, 1, 0, stream>>>(sorted);

  err = random_sample_infinilm_detail::InclusiveSumCub<kDev>(
      cub_workspace, workspace_size, sorted, static_cast<int>(n), stream);
  assert(err == 0 && "`RandomSampleInfinilm` CUB inclusive sum failed");

  random_sample_infinilm_detail::RandomSampleInfinilmKernel<kDev>
      <<<1, 1, 0, stream>>>(result, sorted, indices_out, n, random_val, topp,
                            static_cast<std::size_t>(topk));
}

}  // namespace infini::ops

#endif
