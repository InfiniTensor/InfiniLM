#ifndef INFINI_OPS_CUDA_CONVOLUTION_KERNEL_H_
#define INFINI_OPS_CUDA_CONVOLUTION_KERNEL_H_

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <vector>

#include "base/conv1d.h"
#include "base/conv2d.h"
#include "base/conv3d.h"
#include "base/convolution.h"
#include "common/generic_utils.h"
#include "data_type.h"
#include "dispatcher.h"
#include "native/cuda/kernel_commons.cuh"
#include "native/cuda/ops/convolution/kernel.cuh"
#include "native/cuda/runtime_utils.h"

namespace infini::ops {

template <typename Backend, typename Op>
class CudaConv : public Op {
 public:
  using Op::operator();

  template <typename Padding>
  CudaConv(const Tensor input, const Tensor weight, std::optional<Tensor> bias,
           const std::vector<int64_t> stride, const Padding padding,
           const std::vector<int64_t> dilation, const int64_t groups,
           Tensor out)
      : Op{input, weight, bias, stride, padding, dilation, groups, out} {
    InitializeMetadata();
  }

  CudaConv(const Tensor input, const Tensor weight, std::optional<Tensor> bias,
           const std::vector<int64_t> stride,
           const std::vector<int64_t> padding,
           const std::vector<int64_t> dilation, const bool transposed,
           const std::vector<int64_t> output_padding, const int64_t groups,
           Tensor out)
      : Op{input,    weight,     bias,           stride, padding,
           dilation, transposed, output_padding, groups, out} {
    InitializeMetadata();
  }

  ~CudaConv() { Backend::Free(d_metadata_); }

  void operator()(const Tensor input, const Tensor weight,
                  std::optional<Tensor> bias,
                  const std::vector<int64_t> /*stride*/,
                  const std::vector<int64_t> /*padding*/,
                  const std::vector<int64_t> /*dilation*/,
                  const int64_t /*groups*/, Tensor out) const {
    const auto& metadata = this->metadata_;
    if (metadata.output_size == 0) {
      return;
    }

    auto cuda_stream = static_cast<typename Backend::Stream>(
        this->stream_ ? this->stream_ : 0);
    int block_size = std::min(
        RuntimeUtils<Backend::kDeviceType>::GetOptimalBlockSize(), 1024);
    dim3 block(
        std::min(static_cast<Tensor::Size>(block_size), metadata.output_size));
    dim3 grid(utils::CeilDiv(metadata.output_size, block.x));

    DispatchFunc<ConcatType<List<DataType::kFloat32>, ReducedFloatTypes>,
                 List<128, 256, 512, 1024>>(
        {static_cast<int64_t>(metadata.out_type), block_size},
        [&](auto list_tag) {
          using T = TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;
          constexpr int kBlockSize = ListGet<1>(list_tag);
          const T* bias_ptr = bias.has_value()
                                  ? reinterpret_cast<const T*>(bias->data())
                                  : nullptr;

          ConvKernel<Backend::kDeviceType, T, kBlockSize>
              <<<grid, block, 0, cuda_stream>>>(
                  reinterpret_cast<T*>(out.data()),
                  reinterpret_cast<const T*>(input.data()),
                  reinterpret_cast<const T*>(weight.data()), bias_ptr,
                  d_input_shape_, d_weight_shape_, d_out_shape_,
                  d_input_strides_, d_weight_strides_, d_out_strides_,
                  d_bias_strides_, d_padding_, d_stride_, d_dilation_,
                  metadata.output_size, metadata.spatial_ndim,
                  metadata.kernel_size, metadata.groups, metadata.has_bias);
        },
        "CudaConv::operator()");
  }

  void operator()(const Tensor input, const Tensor weight,
                  std::optional<Tensor> bias, const std::vector<int64_t> stride,
                  const std::vector<int64_t> padding,
                  const std::vector<int64_t> dilation,
                  const bool /*transposed*/,
                  const std::vector<int64_t> /*output_padding*/,
                  const int64_t groups, Tensor out) const {
    (*this)(input, weight, bias, stride, padding, dilation, groups, out);
  }

 private:
  void InitializeMetadata() {
    const auto& metadata = this->metadata_;
    size_t input_shape_size =
        metadata.input_shape.size() * sizeof(*d_input_shape_);
    size_t weight_shape_size =
        metadata.weight_shape.size() * sizeof(*d_weight_shape_);
    size_t out_shape_size = metadata.out_shape.size() * sizeof(*d_out_shape_);
    size_t input_strides_size =
        metadata.input_strides.size() * sizeof(*d_input_strides_);
    size_t weight_strides_size =
        metadata.weight_strides.size() * sizeof(*d_weight_strides_);
    size_t out_strides_size =
        metadata.out_strides.size() * sizeof(*d_out_strides_);
    size_t bias_strides_size = sizeof(*d_bias_strides_);
    size_t attrs_size = metadata.spatial_ndim * sizeof(*d_padding_);
    const size_t metadata_size = input_shape_size + weight_shape_size +
                                 out_shape_size + input_strides_size +
                                 weight_strides_size + out_strides_size +
                                 bias_strides_size + 3 * attrs_size;
    std::vector<std::byte> host_metadata(metadata_size);

    if (Backend::Malloc((void**)&d_metadata_, metadata_size) != 0 ||
        d_metadata_ == nullptr) {
      if (d_metadata_ != nullptr) {
        Backend::Free(d_metadata_);
        d_metadata_ = nullptr;
      }
      throw std::runtime_error("CudaConv failed to allocate device metadata");
    }

    size_t offset = 0;
    d_input_shape_ = reinterpret_cast<Tensor::Size*>(d_metadata_ + offset);
    std::memcpy(host_metadata.data() + offset, metadata.input_shape.data(),
                input_shape_size);
    offset += input_shape_size;

    d_weight_shape_ = reinterpret_cast<Tensor::Size*>(d_metadata_ + offset);
    std::memcpy(host_metadata.data() + offset, metadata.weight_shape.data(),
                weight_shape_size);
    offset += weight_shape_size;

    d_out_shape_ = reinterpret_cast<Tensor::Size*>(d_metadata_ + offset);
    std::memcpy(host_metadata.data() + offset, metadata.out_shape.data(),
                out_shape_size);
    offset += out_shape_size;

    d_input_strides_ = reinterpret_cast<Tensor::Stride*>(d_metadata_ + offset);
    std::memcpy(host_metadata.data() + offset, metadata.input_strides.data(),
                input_strides_size);
    offset += input_strides_size;

    d_weight_strides_ = reinterpret_cast<Tensor::Stride*>(d_metadata_ + offset);
    std::memcpy(host_metadata.data() + offset, metadata.weight_strides.data(),
                weight_strides_size);
    offset += weight_strides_size;

    d_out_strides_ = reinterpret_cast<Tensor::Stride*>(d_metadata_ + offset);
    std::memcpy(host_metadata.data() + offset, metadata.out_strides.data(),
                out_strides_size);
    offset += out_strides_size;

    d_bias_strides_ = reinterpret_cast<Tensor::Stride*>(d_metadata_ + offset);
    Tensor::Stride bias_stride =
        metadata.has_bias ? metadata.bias_strides[0] : 0;
    std::memcpy(host_metadata.data() + offset, &bias_stride, bias_strides_size);
    offset += bias_strides_size;

    d_padding_ = reinterpret_cast<int64_t*>(d_metadata_ + offset);
    std::memcpy(host_metadata.data() + offset, metadata.padding.data(),
                attrs_size);
    offset += attrs_size;

    d_stride_ = reinterpret_cast<int64_t*>(d_metadata_ + offset);
    std::memcpy(host_metadata.data() + offset, metadata.stride.data(),
                attrs_size);
    offset += attrs_size;

    d_dilation_ = reinterpret_cast<int64_t*>(d_metadata_ + offset);
    std::memcpy(host_metadata.data() + offset, metadata.dilation.data(),
                attrs_size);

    if (Backend::Memcpy(d_metadata_, host_metadata.data(), metadata_size,
                        Backend::kMemcpyHostToDevice) != 0) {
      Backend::Free(d_metadata_);
      d_metadata_ = nullptr;
      throw std::runtime_error("CudaConv failed to upload device metadata");
    }
  }

  std::byte* d_metadata_{nullptr};

  Tensor::Size* d_input_shape_{nullptr};

  Tensor::Size* d_weight_shape_{nullptr};

  Tensor::Size* d_out_shape_{nullptr};

  Tensor::Stride* d_input_strides_{nullptr};

  Tensor::Stride* d_weight_strides_{nullptr};

  Tensor::Stride* d_out_strides_{nullptr};

  Tensor::Stride* d_bias_strides_{nullptr};

  int64_t* d_padding_{nullptr};

  int64_t* d_stride_{nullptr};

  int64_t* d_dilation_{nullptr};
};

}  // namespace infini::ops

#endif
