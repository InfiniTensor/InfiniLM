#include "lfm2_short_conv.hpp"

#include "../../global_state/global_state.hpp"

#include <infinicore/ops/add.hpp>
#include <infinicore/ops/embedding.hpp>
#include <infinicore/ops/index_copy.hpp>
#include <infinicore/ops/matmul.hpp>
#include <infinicore/ops/mul.hpp>
#include <infinicore/ops/sum.hpp>

#include <stdexcept>
#include <utility>

namespace infinilm::models::lfm2 {

Lfm2ShortConv::Lfm2ShortConv(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    size_t layer_idx,
    const infinicore::Device &device)
    : layer_idx_(layer_idx) {
    device_ = device;
    hidden_size_ = model_config->get<size_t>("hidden_size");
    kernel_size_ = model_config->get<size_t>("conv_L_cache");
    use_bias_ = model_config->get_or<bool>("conv_bias", false);
    const auto &dtype = model_config->get_dtype();

    if (kernel_size_ < 2) {
        throw std::runtime_error("Lfm2ShortConv: conv_L_cache must be at least 2");
    }

    INFINICORE_NN_MODULE_INIT(
        in_proj, hidden_size_, 3 * hidden_size_, use_bias_, dtype, device);
    INFINICORE_NN_MODULE_INIT(
        out_proj, hidden_size_, hidden_size_, use_bias_, dtype, device);

    // Keep the nested name `conv.weight` so that the complete parameter name is
    // `model.layers.<i>.conv.conv.weight`, matching the released checkpoint.
    conv_weight_ = infinicore::nn::Parameter(
        {hidden_size_, 1, kernel_size_}, dtype, device);
    this->register_parameter("conv.weight", conv_weight_);
    if (use_bias_) {
        conv_bias_ = infinicore::nn::Parameter({hidden_size_}, dtype, device);
        this->register_parameter("conv.bias", conv_bias_);
    }
}

infinicore::Tensor Lfm2ShortConv::causal_depthwise_conv_(
    const infinicore::Tensor &input) const {
    const auto &shape = input->shape();
    if (shape.size() != 3 || shape[2] != hidden_size_) {
        throw std::runtime_error(
            "Lfm2ShortConv: expected input shape [batch, sequence, hidden_size]");
    }

    size_t batch_size = shape[0];
    size_t sequence_length = shape[1];
    const size_t state_length = kernel_size_ - 1;

    auto &forward_context = infinilm::global_state::get_forward_context();
    auto &metadata = forward_context.mamba_metadata;
    const bool has_state_routing = metadata.init_state_indices.has_value()
                                && metadata.final_state_indices.has_value();

    // Paged decode flattens B one-token requests to [1, B, H]. Restore the
    // logical batch dimension so every request consumes its own cache row.
    infinicore::Tensor convolution_input = input;
    bool restore_flattened_shape = false;
    if (has_state_routing) {
        const size_t request_count = metadata.init_state_indices.value()->numel();
        if (batch_size == 1 && request_count > 1) {
            if (sequence_length != request_count) {
                throw std::runtime_error(
                    "Lfm2ShortConv: packed multi-request prefill is not implemented yet");
            }
            convolution_input = input->view({request_count, 1, hidden_size_});
            batch_size = request_count;
            sequence_length = 1;
            restore_flattened_shape = true;
        } else if (request_count != batch_size) {
            throw std::runtime_error(
                "Lfm2ShortConv: cache index count does not match the logical batch size");
        }
    }

    infinicore::Tensor state;
    bool persist_state = false;
    if (layer_idx_ < forward_context.conv_state_vec.size()
        && forward_context.conv_state_vec[layer_idx_]) {
        state = forward_context.conv_state_vec[layer_idx_];
        persist_state = true;
        if (state->ndim() != 3
            || state->size(0) < batch_size
            || state->size(1) != hidden_size_
            || state->size(2) != state_length) {
            throw std::runtime_error("Lfm2ShortConv: incompatible convolution cache shape");
        }
    } else {
        state = infinicore::Tensor::zeros(
            {batch_size, hidden_size_, state_length},
            input->dtype(),
            input->device());
    }

    // Convert the cache from [B, H, K-1] to time-major [B, K-1, H], then
    // append the current B*x values. This is the explicit left padding used by
    // the reference Conv1d(padding=K-1), but it also works during token decode.
    infinicore::Tensor state_batch;
    if (has_state_routing) {
        auto flat_state = state->view(
            {state->size(0), hidden_size_ * state_length});
        state_batch = infinicore::op::embedding(
                          metadata.init_state_indices.value(), flat_state)
                          ->view({batch_size, hidden_size_, state_length});
    } else {
        state_batch = state->narrow({{0, 0, batch_size}});
    }
    auto history = state_batch->permute({0, 2, 1})->contiguous();
    auto combined = infinicore::Tensor::empty(
        {batch_size, state_length + sequence_length, hidden_size_},
        input->dtype(),
        input->device());
    combined->narrow({{1, 0, state_length}})->copy_from(history);
    combined->narrow({{1, state_length, sequence_length}})
        ->copy_from(convolution_input);
    infinicore::Tensor output;
    const bool low_precision = input->dtype() != infinicore::DataType::F32;
    if (low_precision && sequence_length > 1) {
        // Conv1d prefill accumulates products in F32 and casts once. Express
        // depthwise K-tap dot products as a batch of [S,K] x [K,1] GEMMs.
        // This uses portable InfiniCore ops, not ATen or a CUDA-only cast.
        auto windows = combined->as_strided(
            {batch_size, hidden_size_, sequence_length, kernel_size_},
            {static_cast<ptrdiff_t>((state_length + sequence_length) * hidden_size_),
             1, static_cast<ptrdiff_t>(hidden_size_), static_cast<ptrdiff_t>(hidden_size_)})
                           ->contiguous()
                           ->view({batch_size * hidden_size_, sequence_length, kernel_size_});
        auto kernels = conv_weight_->view({hidden_size_, kernel_size_, 1})
                           ->as_strided({batch_size, hidden_size_, kernel_size_, 1},
                                        {0, static_cast<ptrdiff_t>(kernel_size_), 1, 1})
                           ->contiguous()
                           ->view({batch_size * hidden_size_, kernel_size_, 1});
        output = infinicore::op::matmul(windows, kernels)
                     ->view({batch_size, hidden_size_, sequence_length})
                     ->permute({0, 2, 1})->contiguous();
    } else {
        auto terms = low_precision ? infinicore::Tensor::empty(
            {kernel_size_, batch_size, sequence_length, hidden_size_},
            input->dtype(), input->device()) : infinicore::Tensor{};
        for (size_t kernel_idx = 0; kernel_idx < kernel_size_; ++kernel_idx) {
            auto input_window = combined->narrow(
                {{1, kernel_idx, sequence_length}});
            auto kernel_vector = conv_weight_
                                     ->narrow({{2, kernel_idx, 1}})
                                     ->permute({1, 2, 0})
                                     ->contiguous()
                                     ->view({hidden_size_});
            // Mul does not infer a broadcasted output shape. Zero strides
            // expose channel weights as [B,S,H] without B*S weight copies.
            auto expanded_kernel = kernel_vector->as_strided(
                input_window->shape(), {0, 0, 1});
            auto term = infinicore::op::mul(input_window, expanded_kernel);
            if (low_precision) {
                // Reference slow decode rounds products, then sums in F32.
                terms->narrow({{0, kernel_idx, 1}})
                    ->view({batch_size, sequence_length, hidden_size_})->copy_from(term);
            } else {
                output = output ? infinicore::op::add(output, term) : std::move(term);
            }
        }
        if (low_precision) {
            output = infinicore::op::sum(terms, {0});
        }
    }

    if (use_bias_) {
        auto bias = conv_bias_->view({1, 1, hidden_size_});
        output = infinicore::op::add(output, bias);
    }
    if (persist_state) {
        auto newest_history = combined
                                  ->narrow({{1, sequence_length, state_length}})
                                  ->permute({0, 2, 1})
                                  ->contiguous();
        if (has_state_routing) {
            auto flat_state = state->view(
                {state->size(0), hidden_size_ * state_length});
            auto flat_newest = newest_history->view(
                {batch_size, hidden_size_ * state_length});
            infinicore::op::index_copy_(
                flat_state,
                flat_state,
                0,
                metadata.final_state_indices.value(),
                flat_newest);
        } else {
            state_batch->copy_from(newest_history);
        }
    }
    if (restore_flattened_shape) {
        return output->view({1, batch_size, hidden_size_});
    }
    return output;
}

infinicore::Tensor Lfm2ShortConv::forward(
    const infinicore::Tensor &hidden_states) const {
    auto input = hidden_states;
    auto projected = in_proj_->forward(input);

    auto gate_b = projected->narrow({{2, 0, hidden_size_}});
    auto gate_c = projected->narrow({{2, hidden_size_, hidden_size_}});
    auto value = projected->narrow({{2, 2 * hidden_size_, hidden_size_}});

    auto gated_input = infinicore::op::mul(gate_b, value);
    auto convolved = causal_depthwise_conv_(gated_input);
    auto gated_output = infinicore::op::mul(gate_c, convolved);
    return out_proj_->forward(gated_output);
}

} // namespace infinilm::models::lfm2
