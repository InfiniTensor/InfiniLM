#include "deepseek_v4_compressor.hpp"

#include "deepseek_v4_linear.hpp"
#include "deepseek_v4_utils.hpp"
#include "infinicore/ops/deepseek_v4_compressor.hpp"

#include <stdexcept>
#include <vector>

namespace infinilm::models::deepseek_v4 {

DeepseekV4Compressor::DeepseekV4Compressor(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                           size_t compress_ratio,
                                           size_t head_dim,
                                           const infinicore::Device &device)
    : compress_ratio_(compress_ratio),
      head_dim_(head_dim),
      coff_(compress_ratio == 4 ? 2 : 1),
      rms_norm_eps_(model_config->get<double>("rms_norm_eps")) {
    const auto &dtype = model_config->get_dtype();
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t compressed_dim = coff_ * head_dim;

    auto none_quantization = deepseek_v4_linear_quantization(model_config, false);
    INFINICORE_NN_PARAMETER_INIT(ape, ({compress_ratio, compressed_dim}, dtype, device));
    INFINICORE_NN_MODULE_INIT(wkv, hidden_size, compressed_dim, none_quantization, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(wgate, hidden_size, compressed_dim, none_quantization, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(norm, head_dim, rms_norm_eps_, dtype, device);
}

void DeepseekV4Compressor::process_weights_after_loading() {
    if (ape_converted_ || coff_ != 2) {
        return;
    }

    auto ape = tensor_to_float_vector(ape_);
    const size_t compressed_dim = coff_ * head_dim_;
    if (ape.size() != compress_ratio_ * compressed_dim) {
        throw std::runtime_error("DeepseekV4Compressor: unexpected APE shape");
    }

    std::vector<float> converted(ape.size());
    for (size_t row = 0; row < compress_ratio_; ++row) {
        for (size_t col = 0; col < compressed_dim; ++col) {
            const size_t flat = row * compressed_dim + col;
            const size_t cat_row = flat / head_dim_;
            const size_t cat_col = flat % head_dim_;
            const bool second_half = cat_row >= compress_ratio_;
            const size_t src_row = second_half ? cat_row - compress_ratio_ : cat_row;
            const size_t src_col = (second_half ? head_dim_ : 0) + cat_col;
            converted[flat] = ape[src_row * compressed_dim + src_col];
        }
    }

    auto converted_tensor = float_vector_to_tensor(converted, ape_->shape(), ape_->dtype(), ape_->device());
    ape_->copy_from(converted_tensor);
    ape_converted_ = true;
}

infinicore::Tensor DeepseekV4Compressor::forward_tensor(const infinicore::Tensor &hidden_states,
                                                        size_t &batch_size,
                                                        size_t &num_blocks) const {
    if (hidden_states->device().getType() == infinicore::Device::Type::CPU) {
        throw std::runtime_error("DeepseekV4Compressor: GPU tensor required");
    }

    const auto shape = hidden_states->shape();
    batch_size = shape[0];
    const size_t seq_len = shape[1];
    const size_t usable_len = (seq_len / compress_ratio_) * compress_ratio_;
    num_blocks = usable_len / compress_ratio_;
    if (num_blocks == 0) {
        return infinicore::Tensor::empty({batch_size, 0, head_dim_}, hidden_states->dtype(), hidden_states->device());
    }
    auto hidden_states_mut = hidden_states;
    auto kv_t = wkv_->forward(hidden_states_mut);
    auto score_t = wgate_->forward(hidden_states_mut);
    return infinicore::op::deepseek_v4_compressor(kv_t->contiguous(),
                                                  score_t->contiguous(),
                                                  ape_->contiguous(),
                                                  norm_->weight()->contiguous(),
                                                  compress_ratio_,
                                                  static_cast<float>(rms_norm_eps_));
}

} // namespace infinilm::models::deepseek_v4
