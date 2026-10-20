#ifndef INFINI_OPS_CUDA_EMBEDDING_KERNEL_H_
#define INFINI_OPS_CUDA_EMBEDDING_KERNEL_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "base/embedding.h"
#include "common/generic_utils.h"
#include "data_type.h"
#include "dispatcher.h"
#include "native/cuda/device_buffer.h"
#include "native/cuda/kernel_commons.cuh"
#include "native/cuda/ops/embedding/kernel.cuh"

namespace infini::ops {

template <typename Backend>
class CudaEmbedding : public Embedding {
 public:
  CudaEmbedding(const Tensor input, const Tensor weight,
                const std::optional<int64_t> padding_idx,
                const std::optional<double> max_norm, const double norm_type,
                const bool scale_grad_by_freq, const bool sparse, Tensor out)
      : Embedding{input,    weight,    padding_idx,
                  max_norm, norm_type, scale_grad_by_freq,
                  sparse,   out},
        input_ndim_{input.ndim()},
        out_ndim_{out.ndim()},
        is_input_contiguous_{input.IsContiguous()},
        is_out_contiguous_{out.IsContiguous()},
        weight_row_stride_{weight.stride(0)},
        weight_col_stride_{weight.stride(1)} {
    const size_t input_shape_size = input_ndim_ * sizeof(*d_input_shape_);
    const size_t input_strides_size = input_ndim_ * sizeof(*d_input_strides_);
    const size_t out_shape_size = out_ndim_ * sizeof(*d_out_shape_);
    const size_t out_strides_size = out_ndim_ * sizeof(*d_out_strides_);
    const size_t metadata_size = input_shape_size + input_strides_size +
                                 out_shape_size + out_strides_size;
    if (num_indices_ != 0 && !(is_input_contiguous_ && is_out_contiguous_)) {
      std::vector<std::byte> metadata(metadata_size);
      auto* d_metadata_ = metadata_.Allocate(metadata_size);
      size_t offset = 0;
      d_input_shape_ = reinterpret_cast<Tensor::Size*>(d_metadata_ + offset);
      std::memcpy(metadata.data() + offset, input_shape_.data(),
                  input_shape_size);
      offset += input_shape_size;
      d_input_strides_ =
          reinterpret_cast<Tensor::Stride*>(d_metadata_ + offset);
      std::memcpy(metadata.data() + offset, input_strides_.data(),
                  input_strides_size);
      offset += input_strides_size;
      d_out_shape_ = reinterpret_cast<Tensor::Size*>(d_metadata_ + offset);
      std::memcpy(metadata.data() + offset, out_shape_.data(), out_shape_size);
      offset += out_shape_size;
      d_out_strides_ = reinterpret_cast<Tensor::Stride*>(d_metadata_ + offset);
      std::memcpy(metadata.data() + offset, out_strides_.data(),
                  out_strides_size);
      if (Backend::Memcpy(d_metadata_, metadata.data(), metadata_size,
                          Backend::kMemcpyHostToDevice) != Backend::kSuccess)
        throw std::runtime_error("CudaEmbedding metadata copy failed");
    }
    if (max_norm.has_value() && vocab_size_ > 0) {
      d_visited_ = reinterpret_cast<int*>(
          visited_.Allocate(vocab_size_ * sizeof(*d_visited_)));
    }
  }

  CudaEmbedding(const Tensor input, const Tensor weight, Tensor out)
      : CudaEmbedding(input, weight, std::nullopt, std::nullopt, 2.0, false,
                      false, out) {}

  /// \deprecated Use the overload that also accepts `max_norm` and
  /// `norm_type` instead.
  [[deprecated("Use the PyTorch-compatible overload instead.")]]
  CudaEmbedding(const Tensor input, const Tensor weight,
                const int64_t padding_idx, const bool scale_grad_by_freq,
                const bool sparse, Tensor out)
      : CudaEmbedding(input, weight, padding_idx, std::nullopt, 2.0,
                      scale_grad_by_freq, sparse, out) {}

  void operator()(const Tensor input, const Tensor weight,
                  const std::optional<int64_t> /*padding_idx*/,
                  const std::optional<double> max_norm, const double norm_type,
                  const bool /*scale_grad_by_freq*/, const bool /*sparse*/,
                  Tensor out) const override {
    if (num_indices_ == 0) {
      return;
    }

    auto cuda_stream =
        static_cast<typename Backend::Stream>(stream_ ? stream_ : 0);

    constexpr size_t kRenormBlockSize = 256;
    if (max_norm.has_value()) {
      assert(d_visited_ && "`CudaEmbedding` renorm state is not initialized");
      const size_t clear_grid_size =
          utils::CeilDiv(vocab_size_, kRenormBlockSize);
      ClearEmbeddingVisitedKernel<kRenormBlockSize>
          <<<clear_grid_size, kRenormBlockSize, 0, cuda_stream>>>(d_visited_,
                                                                  vocab_size_);
    }

    size_t grid_size = utils::CeilDiv(num_indices_, block_size_);

    DispatchFunc<List<DataType::kInt32, DataType::kInt64>,
                 ConcatType<List<DataType::kFloat32>, ReducedFloatTypes>>(
        {static_cast<int64_t>(input_dtype_),
         static_cast<int64_t>(weight_dtype_)},
        [&](auto list_tag) {
          using IndexT =
              TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;
          using T = TypeMapType<Backend::kDeviceType, ListGet<1>(list_tag)>;

          if (max_norm.has_value()) {
            EmbeddingRenormKernel<kRenormBlockSize, Backend::kDeviceType, T,
                                  IndexT>
                <<<num_indices_, kRenormBlockSize, 0, cuda_stream>>>(
                    reinterpret_cast<T*>(const_cast<void*>(weight.data())),
                    reinterpret_cast<const IndexT*>(input.data()), d_visited_,
                    num_indices_, input_ndim_, d_input_shape_, d_input_strides_,
                    weight_row_stride_, weight_col_stride_, embedding_dim_,
                    vocab_size_, is_input_contiguous_, *max_norm, norm_type);
          }

          EmbeddingKernel<Backend::kDeviceType, T, IndexT>
              <<<grid_size, block_size_, 0, cuda_stream>>>(
                  reinterpret_cast<T*>(out.data()),
                  reinterpret_cast<const IndexT*>(input.data()),
                  reinterpret_cast<const T*>(weight.data()), num_indices_,
                  input_ndim_, d_input_shape_, d_input_strides_, out_ndim_,
                  d_out_shape_, d_out_strides_, weight_row_stride_,
                  weight_col_stride_, embedding_dim_, vocab_size_,
                  is_input_contiguous_, is_out_contiguous_);
        },
        "CudaEmbedding::operator()");
  }

 private:
  const size_t block_size_ =
      embedding_dim_ <= 64 ? 512 : (embedding_dim_ >= 1024 ? 128 : 256);
  Tensor::Size input_ndim_{0};

  Tensor::Size out_ndim_{0};

  bool is_input_contiguous_{false};

  bool is_out_contiguous_{false};

  Tensor::Stride weight_row_stride_{0};

  Tensor::Stride weight_col_stride_{0};

  int* d_visited_{nullptr};

  CudaDeviceBuffer<Backend> metadata_;
  CudaDeviceBuffer<Backend> visited_;

  Tensor::Size* d_input_shape_{nullptr};

  Tensor::Stride* d_input_strides_{nullptr};

  Tensor::Size* d_out_shape_{nullptr};

  Tensor::Stride* d_out_strides_{nullptr};
};

}  // namespace infini::ops

#endif
