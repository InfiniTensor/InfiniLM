#include "infinicore/ops/affine_grid.hpp"
#include <stdexcept>
#include <string>
#include <vector>

namespace infinicore::op {

common::OpDispatcher<AffineGrid::schema> &AffineGrid::dispatcher() {
    static common::OpDispatcher<AffineGrid::schema> dispatcher_;
    return dispatcher_;
};

void AffineGrid::execute(Tensor output, Tensor theta, bool align_corners) {
    auto device_type = context::getDevice().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error("No AffineGrid implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    func(output, theta, align_corners);
}

Tensor affine_grid(Tensor theta, const std::vector<int64_t> &size, bool align_corners) {
    if (theta->ndim() != 3) {
        throw std::runtime_error("AffineGrid: Theta tensor must be 3D (N, 2, 3).");
    }
    if (theta->shape()[1] != 2 || theta->shape()[2] != 3) {
        throw std::runtime_error("AffineGrid: Theta tensor shape must be (N, 2, 3).");
    }

    if (size.size() != 4) {
        throw std::runtime_error("AffineGrid: target size length must be 4 (N, C, H, W).");
    }

    if (static_cast<int64_t>(theta->shape()[0]) != size[0]) {
        throw std::runtime_error("AffineGrid: Theta batch size does not match target size batch.");
    }

    if (!theta->is_contiguous()) {
        theta = theta->contiguous();
    }

    std::vector<size_t> out_shape;
    out_shape.reserve(4);
    out_shape.push_back(static_cast<size_t>(size[0]));
    out_shape.push_back(static_cast<size_t>(size[2]));
    out_shape.push_back(static_cast<size_t>(size[3]));
    out_shape.push_back(2);

    auto output = Tensor::empty(out_shape, theta->dtype(), theta->device());

    AffineGrid::execute(output, theta, align_corners);

    return output;
}

} // namespace infinicore::op
