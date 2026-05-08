#include "infinicore/nn/parameter.hpp"

#include "infinicore/context/context.hpp"

#include <cstring>
#include <stdexcept>

namespace infinicore::nn {
Parameter::Parameter()
    : Tensor() {
}

inline Shape get_partition_shape_(const Shape &shape, Size tp_dim, Size tp_size, Size num_shards) {
    if (tp_size <= 1) {
        return shape;
    }
    Shape part_shape = shape;
    if (tp_dim < shape.size()) {
        Size partition_factor = (num_shards > 0) ? num_shards : tp_size;
        if (shape[tp_dim] % partition_factor != 0) {
            throw std::runtime_error("Tensor dimension " + std::to_string(tp_dim) + " with size " + std::to_string(shape[tp_dim]) + " is not divisible by " + (num_shards > 0 ? "num_shards " : "tp_size ") + std::to_string(partition_factor) + ".");
        }
        part_shape[tp_dim] = shape[tp_dim] / partition_factor;
    }
    return part_shape;
}

Parameter::Parameter(const Tensor &tensor, Size tp_dim, Size tp_rank, Size tp_size, Size num_shards) : Tensor(tensor), tp_dim_(tp_dim), tp_rank_(tp_rank), tp_size_(tp_size), num_shards_(num_shards) {
    if (tp_rank_ >= tp_size_) {
        throw std::runtime_error("Tensor parallel rank " + std::to_string(tp_rank_) + " must be less than tensor parallel size " + std::to_string(tp_size_) + ".");
    }
}

Parameter::Parameter(
    const Shape &shape,
    const DataType &dtype,
    const Device &device,
    Size tp_dim,
    Size tp_rank,
    Size tp_size,
    Size num_shards)
    : Parameter(Tensor::empty(get_partition_shape_(shape, tp_dim, tp_size, num_shards), dtype, device, false), tp_dim, tp_rank, tp_size, num_shards) {
}

Parameter::Parameter(const Parameter &other)
    : Tensor(other),
      tp_dim_(other.tp_dim_),
      tp_rank_(other.tp_rank_),
      tp_size_(other.tp_size_),
      num_shards_(other.num_shards_) {}

void Parameter::load_blob(const void *data) {
    Shape expected_shape = Shape(impl_->shape());
    expected_shape[tp_dim_] *= tp_size_;
    auto buffer = Tensor::empty(expected_shape, impl_->dtype(), Device(Device::Type::CPU, 0), true);
    std::memcpy(buffer->data(), data, buffer->nbytes());
    this->load(buffer);
}

void Parameter::load(const Tensor &tensor) {
    if (impl_->dtype() != tensor->dtype()) {
        throw std::runtime_error("Dtype mismatch when loading tensor into parameter. Weight: " + impl_->info() + ", Tensor: " + tensor->info() + ".");
    }

    Shape expected_shape = Shape(impl_->shape());

    if (num_shards_ == 0 || num_shards_ >= tp_size_) {
        expected_shape[tp_dim_] *= tp_size_;

        if (expected_shape != tensor->shape()) {
            throw std::runtime_error("Shape mismatch when loading tensor into parameter. Weight: " + impl_->info() + ", Tensor: " + tensor->info() + ".");
        }
        if (tp_size_ > 1) {
            impl_->copy_from(tensor->narrow({{tp_dim_, tp_rank_ * impl_->size(tp_dim_), impl_->size(tp_dim_)}}));
        } else {
            impl_->copy_from(tensor);
        }
    } else {
        if (num_shards_ == 0) {
            throw std::runtime_error("num_shards_ is 0 but entered new logic branch!");
        }

        Size replica_size = tp_size_ / num_shards_;
        if (replica_size == 0) {
            throw std::runtime_error("replica_size is 0! tp_size_=" + std::to_string(tp_size_) + ", num_shards_=" + std::to_string(num_shards_));
        }

        Size shard_id = tp_rank_ / replica_size;
        Size shard_size = impl_->size(tp_dim_);
        Size offset = shard_id * shard_size;

        expected_shape[tp_dim_] *= num_shards_;

        if (offset + shard_size > tensor->shape()[tp_dim_]) {
            throw std::runtime_error("Slice out of bounds! offset=" + std::to_string(offset) + ", shard_size=" + std::to_string(shard_size) + ", tensor_dim=" + std::to_string(tensor->shape()[tp_dim_]));
        }

        impl_->copy_from(tensor->narrow({{tp_dim_, offset, shard_size}}));
    }

    infinicore::context::syncStream();
}
} // namespace infinicore::nn
