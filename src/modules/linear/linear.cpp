#include "linear.hpp"

#include "../../context/inference_context.hpp"

namespace infinicore::nn::module {
std::shared_ptr<Linear> Linear::init(
    size_t in_features,
    size_t out_features,
    infiniDtype_t dtype,
    bool has_bias,
    bool weight_transposed,
    weights::DistributionType dist_type,
    int nranks) {
    auto linear = std::shared_ptr<Linear>(new Linear());
    auto out_dim = out_features;
    auto in_dim = in_features;
    linear->nranks = nranks;
    linear->dist_type = dist_type;
    if (dist_type == weights::DistributionType::ROW) {
        in_dim = in_features / nranks;
    } else if (dist_type == weights::DistributionType::COLUMN) {
        out_dim = out_features / nranks;
    }
    if (weight_transposed) {
        linear->weight = Tensor::weight(nullptr, dtype, {in_dim, out_dim})->permute({1, 0});
    } else {
        linear->weight = Tensor::weight(nullptr, dtype, {out_dim, in_dim});
    }
    if (has_bias) {
        linear->bias = Tensor::weight(nullptr, dtype, {out_dim});
    }
    linear->dist_type = dist_type;
    return linear;
}

void Linear::register_weights(infinicore::weights::Loader &loader, const std::string &name_prefix, int rank) {
    loader.register_weight(name_prefix + ".weight", this->weight, rank, this->dist_type);
    if (this->bias != nullptr) {
        if (this->dist_type == weights::DistributionType::COLUMN) {
            loader.register_weight(name_prefix + ".bias", this->bias, rank, this->dist_type);
        } else {
            loader.register_weight(name_prefix + ".bias", this->bias, rank, weights::DistributionType::FULL);
        }
    }
}

void Linear::forward(std::shared_ptr<Tensor> output, std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> residual) {
    getInferenceContext().linear(output, input, this->weight->permute({1, 0}), 1.0, 0.0, residual, this->bias);
}
} // namespace infinicore::nn::module