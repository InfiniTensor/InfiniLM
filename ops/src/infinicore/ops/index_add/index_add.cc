#include "infinicore/ops/index_add.hpp"
#include "infinicore/tensor.hpp"
#include <stdexcept>
#include <string>
#include <vector>

namespace infinicore::op {

common::OpDispatcher<IndexAdd::schema> &IndexAdd::dispatcher() {
    static common::OpDispatcher<IndexAdd::schema> dispatcher_;
    return dispatcher_;
};

void IndexAdd::execute(Tensor output, Tensor input, int64_t dim, Tensor index, Tensor source, float alpha) {
    auto device_type = context::getDevice().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error("No IndexAdd implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    func(output, input, dim, index, source, alpha);
}

static void check_index_add_args(const Tensor &input, int64_t &dim, const Tensor &index, const Tensor &source) {
    int64_t ndim = static_cast<int64_t>(input->ndim());

    if (dim < 0) {
        dim += ndim;
    }
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("IndexAdd: Dimension out of range.");
    }

    if (index->ndim() != 1) {
        throw std::runtime_error("IndexAdd: Index tensor must be 1D.");
    }

    // 使用 DataType::I64 和 I32
    if (index->dtype() != DataType::I64 && index->dtype() != DataType::I32) {
        throw std::runtime_error("IndexAdd: Index tensor must be I32 or I64.");
    }

    if (source->ndim() != input->ndim()) {
        throw std::runtime_error("IndexAdd: Source tensor must have same number of dimensions as input tensor.");
    }

    auto in_shape = input->shape();
    auto src_shape = source->shape();
    auto idx_len = index->shape()[0];

    for (int64_t i = 0; i < ndim; ++i) {
        if (i == dim) {
            if (src_shape[i] != idx_len) {
                throw std::runtime_error("IndexAdd: Source dimension mismatch.");
            }
        } else {
            if (src_shape[i] != in_shape[i]) {
                throw std::runtime_error("IndexAdd: Source non-index dimension mismatch.");
            }
        }
    }
}

// 1. Out-of-place 接口
Tensor index_add(Tensor input, int64_t dim, Tensor index, Tensor source, float alpha) {
    check_index_add_args(input, dim, index, source);

    Tensor output = Tensor::empty(input->shape(), input->dtype(), input->device());
    output->copy_from(input);
    if (!index->is_contiguous()) {
        index = index->contiguous();
    }
    if (!source->is_contiguous()) {
        source = source->contiguous();
    }
    IndexAdd::execute(output, output, dim, index, source, alpha);

    return output;
}

// 2. In-place 接口
void index_add_(Tensor output, Tensor input, int64_t dim, Tensor index, Tensor source, float alpha) {
    check_index_add_args(input, dim, index, source);

    if (output->shape() != input->shape()) {
        throw std::runtime_error("IndexAdd (In-place): Output shape must match Input shape.");
    }

    if (output.operator->() != input.operator->()) {
        output->copy_from(input);
    }

    if (!index->is_contiguous()) {
        index = index->contiguous();
    }
    if (!source->is_contiguous()) {
        source = source->contiguous();
    }

    if (!output->is_contiguous()) {

        Tensor contiguous_out = output->contiguous();

        IndexAdd::execute(contiguous_out, contiguous_out, dim, index, source, alpha);

        output->copy_from(contiguous_out);
    } else {
        // 正常路径: Output 已经是连续的，直接原地执行
        IndexAdd::execute(output, input, dim, index, source, alpha);
    }
}

} // namespace infinicore::op
