#include "infinicore/ops/interpolate.hpp"
#include "../../utils.hpp"

#include <cmath>

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Interpolate);

Interpolate::Interpolate(Tensor out,
                         const Tensor &input,
                         std::string mode,
                         std::vector<int64_t> size,
                         std::vector<double> scale_factor,
                         int align_corners) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, input);
    INFINICORE_GRAPH_OP_DISPATCH(out->device().getType(), out, input, std::move(mode), std::move(size), std::move(scale_factor), align_corners);
}

void Interpolate::execute(Tensor out,
                          const Tensor &input,
                          std::string mode,
                          std::vector<int64_t> size,
                          std::vector<double> scale_factor,
                          int align_corners) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Interpolate, out, input, std::move(mode), std::move(size), std::move(scale_factor), align_corners);
}

static std::vector<size_t> infer_interpolate_shape(
    const std::vector<size_t> &input_shape,
    const std::vector<int64_t> &size,
    const std::vector<double> &scale_factor) {
    if (input_shape.size() < 3) {
        throw std::runtime_error("interpolate expects input with at least 3 dimensions");
    }

    const size_t spatial_ndim = input_shape.size() - 2;
    std::vector<size_t> out_shape = input_shape;

    const bool has_size = !size.empty();
    const bool has_scale = !scale_factor.empty();
    if (has_size == has_scale) {
        throw std::runtime_error("interpolate expects exactly one of size or scale_factor");
    }

    if (has_size) {
        if (size.size() != spatial_ndim) {
            throw std::runtime_error("interpolate size dimensionality mismatch");
        }
        for (size_t i = 0; i < spatial_ndim; ++i) {
            if (size[i] < 0) {
                throw std::runtime_error("interpolate size values must be non-negative");
            }
            out_shape[i + 2] = static_cast<size_t>(size[i]);
        }
        return out_shape;
    }

    if (scale_factor.size() != spatial_ndim) {
        throw std::runtime_error("interpolate scale_factor dimensionality mismatch");
    }
    for (size_t i = 1; i < spatial_ndim; ++i) {
        if (scale_factor[i] != scale_factor[0]) {
            throw std::runtime_error("interpolate only supports scalar/uniform scale_factor");
        }
    }
    const double scale = scale_factor[0];
    if (!std::isfinite(scale) || scale < 0.0) {
        throw std::runtime_error("interpolate scale_factor must be finite and non-negative");
    }
    for (size_t i = 0; i < spatial_ndim; ++i) {
        out_shape[i + 2] = static_cast<size_t>(static_cast<double>(input_shape[i + 2]) * scale);
    }
    return out_shape;
}

static void normalize_interpolate_params(
    const std::vector<size_t> &input_shape,
    std::vector<int64_t> &size,
    std::vector<double> &scale_factor) {
    if (input_shape.size() < 3) {
        throw std::runtime_error("interpolate expects input with at least 3 dimensions");
    }

    const size_t spatial_ndim = input_shape.size() - 2;
    const bool has_size = !size.empty();
    const bool has_scale = !scale_factor.empty();
    if (has_size == has_scale) {
        throw std::runtime_error("interpolate expects exactly one of size or scale_factor");
    }

    if (has_size) {
        if (size.size() == 1 && spatial_ndim > 1) {
            size.assign(spatial_ndim, size[0]);
        }
        if (size.size() != spatial_ndim) {
            throw std::runtime_error("interpolate size dimensionality mismatch");
        }
        for (size_t i = 0; i < spatial_ndim; ++i) {
            if (size[i] < 0) {
                throw std::runtime_error("interpolate size values must be non-negative");
            }
        }
        return;
    }

    if (scale_factor.size() == 1 && spatial_ndim > 1) {
        scale_factor.assign(spatial_ndim, scale_factor[0]);
    }
    if (scale_factor.size() != spatial_ndim) {
        throw std::runtime_error("interpolate scale_factor dimensionality mismatch");
    }
    for (size_t i = 1; i < spatial_ndim; ++i) {
        if (scale_factor[i] != scale_factor[0]) {
            throw std::runtime_error("interpolate only supports scalar/uniform scale_factor");
        }
    }
    if (!std::isfinite(scale_factor[0]) || scale_factor[0] < 0.0) {
        throw std::runtime_error("interpolate scale_factor must be finite and non-negative");
    }
}

Tensor interpolate(const Tensor &input,
                   std::string mode,
                   std::vector<int64_t> size,
                   std::vector<double> scale_factor,
                   int align_corners) {
    normalize_interpolate_params(input->shape(), size, scale_factor);
    auto out_shape = infer_interpolate_shape(input->shape(), size, scale_factor);
    auto out = Tensor::empty(out_shape, input->dtype(), input->device());
    interpolate_(out, input, std::move(mode), std::move(size), std::move(scale_factor), align_corners);
    return out;
}

void interpolate_(Tensor out,
                  const Tensor &input,
                  std::string mode,
                  std::vector<int64_t> size,
                  std::vector<double> scale_factor,
                  int align_corners) {
    normalize_interpolate_params(input->shape(), size, scale_factor);
    Interpolate::execute(out, input, std::move(mode), std::move(size), std::move(scale_factor), align_corners);
}

} // namespace infinicore::op
