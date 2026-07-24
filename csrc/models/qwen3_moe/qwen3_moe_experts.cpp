#include "qwen3_moe_experts.hpp"

#include "../../global_state/global_state.hpp"
#include "../../utils.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/distributed/allreduce.hpp"

#include <cstdint>
#include <cstring>
#include <vector>

namespace infinilm::models::qwen3_moe {

Qwen3MoeExperts::Qwen3MoeExperts(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                 const infinicore::Device &device) {

    dtype_ = model_config->get_dtype();
    device_ = device;

    num_experts_ = model_config->get<size_t>("num_experts");
    num_experts_per_tok_ = model_config->get<size_t>("num_experts_per_tok");
    hidden_size_ = model_config->get<size_t>("hidden_size");
    size_t moe_intermediate_size = model_config->get<size_t>("moe_intermediate_size");

    ASSERT((num_experts_ > 0) && (num_experts_per_tok_ > 0) && (num_experts_per_tok_ <= num_experts_));

    const engine::distributed::RankInfo &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    tp_rank_ = rank_info.tp_rank;
    tp_size_ = rank_info.tp_size;
    communicator_ = rank_info.comm;
    ASSERT(moe_intermediate_size % size_t(tp_size_) == 0);
    moe_intermediate_size_per_rank_ = moe_intermediate_size / size_t(tp_size_);

    gate_proj_stacked_ = infinicore::nn::Parameter(
        {num_experts_, moe_intermediate_size_per_rank_, hidden_size_},
        dtype_, device_);
    up_proj_stacked_ = infinicore::nn::Parameter(
        {num_experts_, moe_intermediate_size_per_rank_, hidden_size_},
        dtype_, device_);
    down_proj_stacked_ = infinicore::nn::Parameter(
        {num_experts_, hidden_size_, moe_intermediate_size_per_rank_},
        dtype_, device_);

    for (size_t i = 0; i < num_experts_; ++i) {
        const std::string suffix = std::to_string(i) + ".";

        auto gate_slab = gate_proj_stacked_->narrow({{0, i, 1}})->squeeze(0);
        this->register_parameter(suffix + "gate_proj.weight",
                                 infinicore::nn::Parameter(gate_slab, /*tp_dim=*/0, tp_rank_, tp_size_));

        auto up_slab = up_proj_stacked_->narrow({{0, i, 1}})->squeeze(0);
        this->register_parameter(suffix + "up_proj.weight",
                                 infinicore::nn::Parameter(up_slab, /*tp_dim=*/0, tp_rank_, tp_size_));

        auto down_slab = down_proj_stacked_->narrow({{0, i, 1}})->squeeze(0);
        this->register_parameter(suffix + "down_proj.weight",
                                 infinicore::nn::Parameter(down_slab, /*tp_dim=*/1, tp_rank_, tp_size_));
    }
}

infinicore::Tensor Qwen3MoeExperts::forward(const infinicore::Tensor &hidden_states,
                                            const infinicore::Tensor &top_k_index,
                                            const infinicore::Tensor &top_k_weights) const {
    ASSERT(hidden_states->ndim() == 2);
    ASSERT(top_k_index->ndim() == 2);
    ASSERT(top_k_weights->ndim() == 2);

    const size_t ntoken = hidden_states->shape()[0];
    const size_t k = top_k_index->shape()[1];
    const size_t total_assignments = ntoken * k;

    // ---- Step 1: device->host sync of routing tables.
    infinicore::context::syncStream();
    auto top_k_index_cpu = top_k_index->to(infinicore::Device::Type::CPU);
    auto top_k_weights_cpu = top_k_weights->to(infinicore::Device::Type::CPU);
    const int32_t *idx_host = reinterpret_cast<const int32_t *>(top_k_index_cpu->data());
    const float *w_host = reinterpret_cast<const float *>(top_k_weights_cpu->data());

    // ---- Step 2: CPU-side bucketing.
    std::vector<int32_t> counts(num_experts_, 0);
    for (size_t a = 0; a < total_assignments; ++a) {
        int32_t e = idx_host[a];
        ASSERT(e >= 0 && size_t(e) < num_experts_);
        ++counts[e];
    }
    std::vector<int32_t> offsets(num_experts_, 0);
    {
        int32_t acc = 0;
        for (size_t e = 0; e < num_experts_; ++e) {
            offsets[e] = acc;
            acc += counts[e];
        }
    }

    auto perm_token_cpu = infinicore::Tensor::empty(
        {total_assignments}, infinicore::DataType::I32, infinicore::Device(infinicore::Device::Type::CPU));
    auto perm_weight_cpu = infinicore::Tensor::empty(
        {total_assignments}, dtype_, infinicore::Device(infinicore::Device::Type::CPU));
    int32_t *perm_token = reinterpret_cast<int32_t *>(perm_token_cpu->data());
    void *perm_weight_raw = perm_weight_cpu->data();

    {
        std::vector<int32_t> cursor(offsets);
        for (size_t a = 0; a < total_assignments; ++a) {
            int32_t e = idx_host[a];
            int32_t row = cursor[e]++;
            perm_token[row] = int32_t(a / k);
            switch (dtype_) {
            case infinicore::DataType::BF16:
                reinterpret_cast<uint16_t *>(perm_weight_raw)[row] = f32_to_bf16(w_host[a]);
                break;
            case infinicore::DataType::F16:
                reinterpret_cast<uint16_t *>(perm_weight_raw)[row] = f32_to_f16(w_host[a]);
                break;
            case infinicore::DataType::F32:
                reinterpret_cast<float *>(perm_weight_raw)[row] = w_host[a];
                break;
            default:
                throw std::runtime_error("Qwen3MoeExperts: unsupported dtype for routing weight broadcast.");
            }
        }
    }

    auto counts_cpu = infinicore::Tensor::empty(
        {num_experts_}, infinicore::DataType::I32, infinicore::Device(infinicore::Device::Type::CPU));
    std::memcpy(counts_cpu->data(), counts.data(), counts.size() * sizeof(int32_t));

    // ---- Step 3: upload bucketing results to the active device.
    auto perm_token_dev = perm_token_cpu->to(device_);
    auto perm_weight_dev = perm_weight_cpu->to(device_);
    auto counts_dev = counts_cpu->to(device_);

    // ---- Step 4: gather hidden states by permuted token indices.
    auto permuted_hidden = infinicore::op::embedding(perm_token_dev, hidden_states);

    // ---- Step 5: per-projection grouped GEMMs.
    // `counts` already lives on the host (computed in Step 2), so we hand it to
    // grouped_gemm directly: device backends then skip the per-call device->host
    // copy + stream sync of the group sizes. On MetaX that copy is effectively a
    // hard sync, so this removes 3 hard syncs per layer on the decode path.
    // Under graph recording the op runs at replay time, after this local
    // `counts` vector is gone; pass nullptr there so the device-sync path is used.
    const int32_t *counts_host = infinicore::context::isGraphRecording() ? nullptr : counts.data();
    auto gate_out = infinicore::op::grouped_gemm(permuted_hidden, gate_proj_stacked_, counts_dev, 1.0f, 0.0f, counts_host);
    auto up_out = infinicore::op::grouped_gemm(permuted_hidden, up_proj_stacked_, counts_dev, 1.0f, 0.0f, counts_host);
    auto intermediate = infinicore::op::swiglu(up_out, gate_out);
    auto down_out = infinicore::op::grouped_gemm(intermediate, down_proj_stacked_, counts_dev, 1.0f, 0.0f, counts_host);

    // ---- Step 6: scale each row by its routing weight (broadcast over hidden).
    auto weight_broadcast = perm_weight_dev->as_strided(
        {total_assignments, hidden_size_}, {1, 0});
    auto scaled = infinicore::op::mul(down_out, weight_broadcast);

    // ---- Step 7: scatter-add back into a zero-initialized per-token buffer.
    auto output = infinicore::Tensor::zeros({ntoken, hidden_size_}, dtype_, device_);
    infinicore::op::index_add_(output, output, /*dim=*/0, perm_token_dev, scaled, 1.0f);

    // ---- Step 8: TP all-reduce on the partial down_proj output.
    if (tp_size_ > 1 && communicator_ != nullptr) {
        infinicore::op::distributed::allreduce_(output, output, INFINICCL_SUM, communicator_);
        infinicore::context::syncStream();
    }

    return output;
}

} // namespace infinilm::models::qwen3_moe
