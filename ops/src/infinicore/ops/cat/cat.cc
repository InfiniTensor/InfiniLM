#include "infinicore/ops/cat.hpp"
#include "infinicore/context/context.hpp"
#include <cstring>
#include <stdexcept>
namespace infinicore::op {

namespace {

class CatInfo {

    CatInfo() = default;

public:
    int dim;
    int ndim;
    size_t tensors_size;

    std::vector<int> contiguous_dim;
    std::vector<size_t> copy_size;

    static CatInfo create(Tensor &out, std::vector<Tensor> &tensors, int dim) {

        int ndim = out->ndim();
        size_t tensors_size = tensors.size();

        std::vector<int> contiguous_dim(tensors_size, ndim);
        std::vector<size_t> copy_size(tensors_size, dsize(out->dtype()));

        for (int i = 0; i < tensors_size; i++) {
            if (tensors[i]->ndim() == 1) {
                continue;
            }
            if (tensors[i]->stride(ndim - 1) == 1) {
                contiguous_dim[i] = ndim - 1;
                for (int j = ndim - 2; j >= dim; j--) {
                    if (tensors[i]->stride(j) == tensors[i]->stride(j + 1) * tensors[i]->shape()[j + 1]) {
                        contiguous_dim[i] = j;
                    }
                }
            }
            for (int j = contiguous_dim[i]; j < ndim; j++) {
                copy_size[i] *= tensors[i]->shape()[j];
            }
        }

        return CatInfo{dim, ndim, tensors_size, contiguous_dim, copy_size};
    }
};

void low_dim_copy(
    CatInfo &info, Tensor &tensor, Tensor &out,
    std::byte *tensor_ptr, std::byte *out_ptr,
    int depth, int tensor_pos) {

    if (depth != info.contiguous_dim[tensor_pos]) {
        std::byte *now_tensor_ptr = tensor_ptr;
        std::byte *now_out_ptr = out_ptr;

        for (int i = 0; i < tensor->shape()[depth]; i++) {

            low_dim_copy(info, tensor, out, now_tensor_ptr, now_out_ptr, depth + 1, tensor_pos);

            now_tensor_ptr += tensor->stride(depth) * dsize(tensor->dtype());
            now_out_ptr += out->stride(depth) * dsize(out->dtype());
        }
    } else {
        if (out->device().getType() == Device::Type::CPU) {

            std::memcpy(out_ptr, tensor_ptr, info.copy_size[tensor_pos]);
        } else {

            context::memcpyD2D(out_ptr, tensor_ptr, info.copy_size[tensor_pos]);
        }
    }
}

void high_dim_split(
    CatInfo &info, std::vector<Tensor> &tensors, Tensor &out,
    std::vector<std::byte *> tensors_ptr, std::byte *out_ptr,
    int depth) {

    if (depth != info.dim) {
        std::vector<std::byte *> now_tensors_ptr = tensors_ptr;
        std::byte *now_out_ptr = out_ptr;

        for (int i = 0; i < out->shape()[depth]; i++) {

            high_dim_split(info, tensors, out, now_tensors_ptr, now_out_ptr, depth + 1);

            for (int i = 0; i < info.tensors_size; i++) {
                if (tensors[i]->ndim() == 1) {
                    continue;
                }
                now_tensors_ptr[i] += tensors[i]->stride(depth) * dsize(tensors[i]->dtype());
            }
            now_out_ptr += out->stride(depth) * dsize(out->dtype());
        }
    } else {
        std::byte *now_out_ptr = out_ptr;

        for (int i = 0; i < info.tensors_size; i++) {
            if (tensors[i]->ndim() == 1) {
                continue;
            }

            low_dim_copy(info, tensors[i], out, tensors_ptr[i], now_out_ptr, depth, i);

            now_out_ptr += tensors[i]->shape()[depth] * out->stride(depth) * dsize(out->dtype());
        }
    }
}

} // namespace

Tensor cat(std::vector<Tensor> tensors, int dim) {
    assert(tensors.size() >= 2);
    int ndim = tensors[0]->ndim();
    assert(-ndim <= dim && dim < ndim);
    dim = (dim + ndim) % ndim;

    Shape shape = tensors[0]->shape();
    for (int i = 1; i < tensors.size(); i++) {
        assert(tensors[i]->ndim() == dim || tensors[i]->ndim() == 1);
        if (tensors[i]->ndim() != ndim) {
            continue;
        }
        shape[dim] += tensors[i]->shape()[dim];
    }

    auto out = Tensor::empty(shape, tensors[0]->dtype(), tensors[0]->device());
    cat_(out, tensors, dim);
    return out;
}

void cat_(Tensor out, std::vector<Tensor> tensors, int dim) {
    // assert the parameter properties are correct.
    assert(tensors.size() >= 2);
    int ndim = out->ndim();
    assert(-ndim <= dim && dim < ndim);
    dim = (dim + ndim) % ndim;

    size_t dim_shape = 0;
    for (auto &tensor : tensors) {
        assert(tensor->ndim() == ndim || tensors[i]->ndim() == 1);
        if (tensor->ndim() == 1) {
            assert(tensor->shape()[0] == 0);
            continue;
        }
        for (int i = 0; i < ndim; i++) {
            if (i != dim) {
                assert(tensor->shape()[i] == out->shape()[i]);
            } else {
                dim_shape += tensor->shape()[i];
            }
        }
    }
    assert(dim_shape == out->shape()[dim]);

    // Get info
    CatInfo info = CatInfo::create(out, tensors, dim);
    std::vector<std::byte *> tensors_ptr(tensors.size());
    for (int i = 0; i < tensors.size(); i++) {
        tensors_ptr[i] = tensors[i]->data();
    }
    std::byte *out_ptr = out->data();

    high_dim_split(info, tensors, out, tensors_ptr, out_ptr, 0);
}

} // namespace infinicore::op
