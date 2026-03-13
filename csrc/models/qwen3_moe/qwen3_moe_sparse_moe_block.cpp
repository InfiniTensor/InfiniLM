#include "qwen3_moe_sparse_moe_block.hpp"
#include "infinicore/nn/linear.hpp"
#include <cstdio>

#include "infinicore/io.hpp"
namespace infinilm::models::qwen3_moe {

Qwen3MoeSparseMoeBlock::Qwen3MoeSparseMoeBlock(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                               const infinicore::Device &device,
                                               engine::distributed::RankInfo rank_info)
    : model_config_(model_config), rank_info_(rank_info), use_bias_(false) {

    const auto &dtype{model_config_->get_dtype()};

    hidden_size_ = model_config->get<size_t>("hidden_size");
    moe_intermediate_size_ = model_config->get<size_t>("moe_intermediate_size");
    num_experts_ = model_config->get<size_t>("num_experts");
    num_experts_per_tok_ = model_config->get<size_t>("num_experts_per_tok");
    norm_topk_prob_ = model_config->get<bool>("norm_topk_prob");
    INFINICORE_NN_MODULE_INIT(gate, hidden_size_, num_experts_, false, dtype, device);

    experts_.reserve(num_experts_);
    for (size_t i = 0; i < num_experts_; ++i) {
        experts_.push_back(this->register_module<Qwen3MoeMLP>(
            "experts." + std::to_string(i), model_config_, device, rank_info_));
    }
}

infinicore::Tensor Qwen3MoeSparseMoeBlock::forward(const infinicore::Tensor &hidden_states) const {

    // TODO: Implement the forward pass
    auto hidden_states_mutable = hidden_states;
    auto identity = infinicore::op::eye(hidden_size_, hidden_size_, hidden_states->dtype(), hidden_states->device());

    auto shape = hidden_states->shape();
    auto hidden_states_reshape = hidden_states->view({shape[0] * shape[1], shape[2]});
    auto router_states_sum = infinicore::Tensor::empty(hidden_states_reshape->shape(), hidden_states_reshape->dtype(), hidden_states_reshape->device());
    size_t ntoken = hidden_states_reshape->shape()[0]; // shape[ 1 11 2048 ]

    // TopK路由结果缓冲区
    // 专家权重 [ntok * num_experts_per_tok_]
    auto values_gpu = infinicore::Tensor::empty({ntoken, num_experts_per_tok_}, infinicore::DataType::F32, hidden_states->device());
    // 专家索引 [ntok * num_experts_per_tok_]
    auto indices_gpu = infinicore::Tensor::empty({ntoken, num_experts_per_tok_}, infinicore::DataType::I32, hidden_states->device());

    std::vector<float> values_cpu(ntoken * num_experts_per_tok_, 0.f); // CPU端权重（用于后续计算）
    std::vector<int> indices_cpu(ntoken * num_experts_per_tok_, 0);    // CPU端索引（用于后续计算）

    auto router_logits = gate_->forward(hidden_states_reshape);

    infinicore::op::topksoftmax(values_gpu, indices_gpu, router_logits, num_experts_per_tok_, norm_topk_prob_);

    // std::cout << "--------------> xxxxx ::: " << values_cpu.data() << std::endl;
    // std::cout << "--------------> xxxxx ::: " << values_gpu->data() << std::endl;
    // std::cout << "--------------> xxxxx ::: " << values_gpu->numel() << std::endl;
    infinicore::context::memcpyD2H(values_cpu.data(), values_gpu->data(), values_gpu->numel() * sizeof(float));
    infinicore::context::memcpyD2H(indices_cpu.data(), indices_gpu->data(), indices_gpu->numel() * sizeof(int));
    infinicore::context::syncDevice();

    // std::cout << "--------------> xxxxx ::: hidden_states_reshape " << hidden_states_reshape.<< std::endl;

    // std::cout << "--------------> xxxxx ::: values_cpu " << indices_gpu.<< std::endl;

    // std::cout << "--------------> xxxxx ::: indices_cpu " << indices_gpu << std::endl;

    for (size_t itok = 0; itok < ntoken; ++itok) {

        auto hidden_states_reshape_i = hidden_states_reshape->narrow({{0, itok, 1}});

        int index = indices_cpu[itok * num_experts_per_tok_ + 0];
        float alpha = values_cpu[itok * num_experts_per_tok_ + 0];
        infinicore::Tensor router_states_sum_i = experts_[index]->forward(hidden_states_reshape_i); // * alpha
        infinicore::op::matmul_(router_states_sum_i, router_states_sum_i, identity, alpha);

        for (size_t k = 1; k < num_experts_per_tok_; ++k) {
            index = indices_cpu[itok * num_experts_per_tok_ + k];
            alpha = values_cpu[itok * num_experts_per_tok_ + k];
            auto experts_out = experts_[index]->forward(hidden_states_reshape_i);
            infinicore::op::matmul_(experts_out, experts_out, identity, alpha);

            infinicore::op::add_(router_states_sum_i, router_states_sum_i, experts_out);
        }
        router_states_sum->narrow({{0, itok, 1}})->copy_from(router_states_sum_i);
    }

    return router_states_sum->view({shape[0], shape[1], shape[2]});
} // namespace infinilm::models::qwen3_moe

} // namespace infinilm::models::qwen3_moe

// xmake build &&xmake install &&xmake build _infinicore &&xmake install _infinicore
