#include "minicpm5_moe_vllm_fused_sparse_moe_block.hpp"

#include "minicpm5_moe_router_cpu_detail.hpp"
#include "../../utils/nvtx.hpp"
#include "../../utils/vllm_fused_moe_dispatch.hpp"

#include "infinicore/ops/add.hpp"
#include "infinicore/ops/convert_to_f32.hpp"
#include "infinicore/ops/linear.hpp"

#include <cstdlib>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace infinilm::models::minicpm5_moe {

void MiniCPM5MoeVllmFusedSparseMoeBlock::rebuild_stacked_expert_weights(const infinicore::Device &dev,
                                                                        const infinicore::DataType &dt) const {
    if (w1_stacked_.has_value() && w2_stacked_.has_value() && stacked_dev_.has_value() && stacked_dt_.has_value() &&
        stacked_dev_.value() == dev && stacked_dt_.value() == dt) {
        return;
    }

    const size_t E = experts_.size();
    if (E == 0) {
        throw std::runtime_error("MiniCPM5MoeVllmFusedSparseMoeBlock: no experts");
    }
    const size_t inter = experts_[0]->moe_intermediate_size();
    const size_t H = experts_[0]->hidden_size();

    w1_stacked_ = infinicore::Tensor::empty({E, 2 * inter, H}, dt, dev);
    w2_stacked_ = infinicore::Tensor::empty({E, H, inter}, dt, dev);

    for (size_t e = 0; e < E; ++e) {
        auto gate_w = experts_[e]->gate_up_linear()->get_gate_weight()->to(dev)->contiguous();
        auto up_w = experts_[e]->gate_up_linear()->get_up_weight()->to(dev)->contiguous();
        auto down_w = experts_[e]->down_linear()->weight()->to(dev)->contiguous();

        auto w1_plane = w1_stacked_.value()->narrow({{0, e, 1}});
        w1_plane->narrow({{1, 0, inter}})->copy_from(gate_w->unsqueeze(0));
        w1_plane->narrow({{1, inter, inter}})->copy_from(up_w->unsqueeze(0));

        auto w2_plane = w2_stacked_.value()->narrow({{0, e, 1}});
        w2_plane->copy_from(down_w->unsqueeze(0));
    }

    stacked_dev_ = dev;
    stacked_dt_ = dt;
}

infinicore::Tensor MiniCPM5MoeVllmFusedSparseMoeBlock::forward(const infinicore::Tensor &hidden_states) const {
    infinilm::utils::NvtxRange nvtx_moe("MiniCPM5MoeVllmFusedSparseMoeBlock::forward");

    if (const char *dis = std::getenv("INFINILM_DISABLE_VLLM_FUSED_MOE")) {
        if (std::string(dis) == "1") {
            return MiniCPM5MoeSparseMoeBlock::forward(hidden_states);
        }
    }

    // If vLLM isn't available in this interpreter, skip all fused-path prep and use reference MoE.
    const bool fused_avail = infinilm::vllm_fused_moe_dispatch::fused_experts_ic_available();
    if (!fused_avail) {
        if (const char *force_backend = std::getenv("INFINILM_FORCE_MOE_BACKEND")) {
            if (std::string(force_backend) == "vllm_fused") {
                throw std::runtime_error("MiniCPM5MoeVllmFusedSparseMoeBlock: INFINILM_FORCE_MOE_BACKEND=vllm_fused but fused_experts_ic is unavailable");
            }
        }
        return MiniCPM5MoeSparseMoeBlock::forward(hidden_states);
    }

    // The vLLM fused path may require additional temporary allocations (stacked expert weights,
    // topk buffers, torch views, vLLM workspace). If any step fails (OOM / missing deps / Python
    // errors), fall back to the reference per-expert implementation to preserve correctness.
    try {
        auto shape = hidden_states->shape();
        const size_t batch_size = shape[0];
        const size_t seq_len = shape[1];
        const size_t hidden_size = shape[2];
        const size_t n_tokens = batch_size * seq_len;

        const size_t top_k =
            infinilm::global_state::get_infinilm_config().model_config->get<size_t>("num_experts_per_tok");
        const bool norm_topk_prob =
            infinilm::global_state::get_infinilm_config().model_config->get_or<bool>("norm_topk_prob", true);
        const float routed_scaling_factor = static_cast<float>(
            infinilm::global_state::get_infinilm_config().model_config->get<double>("routed_scaling_factor"));
        const size_t n_group =
            infinilm::global_state::get_infinilm_config().model_config->get_or<size_t>("n_group", 1);
        const size_t topk_group =
            infinilm::global_state::get_infinilm_config().model_config->get_or<size_t>("topk_group", 1);

        const size_t n_routed_experts = experts_.size();
        if (n_group == 0 || topk_group == 0 || n_routed_experts == 0 || (n_routed_experts % n_group) != 0) {
            throw std::runtime_error("MiniCPM5MoeVllmFusedSparseMoeBlock: invalid n_group/topk_group/n_routed_experts");
        }

        auto hs2d = hidden_states->view({n_tokens, hidden_size});

        if (!gate_weight_f32_device_.has_value()) {
            gate_weight_f32_device_ = infinicore::op::convert_to_f32(gate_->weight()->contiguous());
        }

        auto hs_f32 = infinicore::op::convert_to_f32(hs2d->contiguous());
        auto router_logits = infinicore::op::linear(hs_f32, gate_weight_f32_device_.value(), std::nullopt);

        const auto &dev = hidden_states->device();
        const auto act_dt = hidden_states->dtype();

        rebuild_stacked_expert_weights(dev, act_dt);

        infinicore::Tensor topk_w_dev;
        infinicore::Tensor topk_ids_dev;
        {
            infinilm::utils::NvtxRange nvtx_cpu_envelope("moe_vllm_fused::router_d2h_cpu_topk_pack_h2d");
            auto logits_cpu = router_logits->to(infinicore::Device::cpu());
            logits_cpu = logits_cpu->contiguous();

            auto bias_cpu = e_score_correction_bias_->to(infinicore::Device::cpu());
            bias_cpu = bias_cpu->contiguous();

            router_cpu_detail::RouterTopkCpuResult router_out;
            router_cpu_detail::run_router_topk_cpu(
                logits_cpu,
                bias_cpu,
                n_tokens,
                n_routed_experts,
                top_k,
                norm_topk_prob,
                routed_scaling_factor,
                n_group,
                topk_group,
                router_out);
            const auto &topk_indices_cpu = router_out.topk_indices;
            const auto &topk_weights_cpu = router_out.topk_weights;

            auto topk_w_cpu = infinicore::Tensor::empty({n_tokens, top_k}, act_dt, infinicore::Device::cpu());
            for (size_t t = 0; t < n_tokens; ++t) {
                for (size_t j = 0; j < top_k; ++j) {
                    router_cpu_detail::write_f32_as_element(topk_w_cpu, t * top_k + j, topk_weights_cpu[t][j]);
                }
            }
            topk_w_dev = topk_w_cpu->to(dev)->contiguous();

            auto topk_ids_cpu =
                infinicore::Tensor::empty({n_tokens, top_k}, infinicore::DataType::I32, infinicore::Device::cpu());
            {
                int32_t *idp = reinterpret_cast<int32_t *>(topk_ids_cpu->data());
                for (size_t t = 0; t < n_tokens; ++t) {
                    for (size_t j = 0; j < top_k; ++j) {
                        idp[t * top_k + j] = topk_indices_cpu[t][j];
                    }
                }
            }
            topk_ids_dev = topk_ids_cpu->to(dev)->contiguous();
        }

        std::optional<infinicore::Tensor> routed_opt;
        {
            infinilm::utils::NvtxRange nvtx_fused("moe_vllm_fused::fused_experts_dispatch");
            routed_opt = infinilm::vllm_fused_moe_dispatch::try_fused_experts_ic(
                hs2d->contiguous(),
                w1_stacked_.value(),
                w2_stacked_.value(),
                topk_w_dev,
                topk_ids_dev);
        }

        if (!routed_opt.has_value()) {
            return MiniCPM5MoeSparseMoeBlock::forward(hidden_states);
        }

        auto routed = routed_opt.value()->view({batch_size, seq_len, hidden_size});
        auto shared = shared_experts_->forward(hidden_states);
        return infinicore::op::add(routed, shared);
    } catch (...) {
        return MiniCPM5MoeSparseMoeBlock::forward(hidden_states);
    }
}

} // namespace infinilm::models::minicpm5_moe

