#pragma once

#include "../quantization/quantization.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/ops.hpp"
#include <infiniccl.h>
#include <optional>

namespace infinilm::nn {

using namespace infinicore::nn;

class BaseLinear : public infinicore::nn::Module {
public:
    BaseLinear(size_t in_features, size_t out_features,
               std::shared_ptr<infinilm::quantization::BaseQuantization> quantization = std::make_shared<infinilm::quantization::NoneQuantization>(nullptr),
               bool bias = true,
               const infinicore::DataType &dtype = infinicore::DataType::F32,
               const infinicore::Device &device = infinicore::Device(),
               int split_dim = -1, int tp_rank = 0, int tp_size = 1,
               int tp_num_heads = -1,
               const std::string &stem = "");

    // Forward pass: output = input @ weight.T + bias
    infinicore::Tensor forward(infinicore::Tensor &input) const;

    // Forward pass with residual connection
    infinicore::Tensor forward(infinicore::Tensor &input, infinicore::Tensor &residual) const;

    // Module information
    size_t in_features() const { return in_features_; }
    size_t out_features() const { return out_features_; }
    bool has_bias() const { return has_bias_; }
    infinicore::DataType dtype() const { return dtype_; }
    float alpha() const { return alpha_; }
    void set_alpha(float alpha) { alpha_ = alpha; }

    // Accessors for parameters (backward compatible)
    infinicore::Tensor weight() const;
    infinicore::Tensor bias() const;
    infinicore::Tensor weight_scale() const;
    infinicore::Tensor weight_zeros() const;
    infinicore::Tensor gidx() const;

    // Get parameter by name
    infinicore::Tensor get_param(const std::string &name) const;

    std::shared_ptr<infinilm::quantization::BaseQuantization> get_quantization() const { return quantization_; }
    void process_weights_after_loading() override;
    void reset_runtime_state() const override;

    // Split fused linear parameters into named sub-parameters
    std::vector<infinilm::quantization::SplitParam> split_params(
        const std::vector<infinilm::quantization::SplitInfo> &splits,
        int tp_rank, int tp_size, int tp_num_heads) const;

    // One shard of a fused linear, for schemes that cannot share a single fused
    // buffer (GGUF block quantization: row_bytes differs per shard type).
    struct FusedShard {
        std::string name;        // "q_proj" / "gate_proj" ... 注册到父模块时用
        size_t out_features;     // 本 shard 的逻辑输出行数
        std::string stem;        // "layers.0.self_attn.q_proj." 类型表查询用
    };

    // 为融合 Linear 逐 shard 各分配一块独立 buffer：本对象 parameters_ 里的 key 是
    // "shard<i>.<suffix>"（i 即输出 dim(-1) 上的顺序），返回值里的 full_name 是
    // "<name>.<suffix>"，交给调用方的 register_fn 注册到父模块（与 split_params 同路）。
    // 只有 get_param_layout(带 stem) 返回空布局（融合组）的方案才走这里。
    // 顺带把每个 shard 的 checkpoint stem 记进 shard_stems_（下标 = 上面的 i）：
    // 组 stem 查不出各 shard 的格式，forward 必须把它们交还给量化方案。
    std::vector<infinilm::quantization::SplitParam> init_fused_shards(
        const std::vector<FusedShard> &shards);

    // Allow subclasses to access the raw parameters map
    const infinicore::nn::Parameter &get_parameter_ref(const std::string &name) const;

protected:
    infinicore::Tensor compute_linear(infinicore::Tensor &input) const;
    infinicore::Tensor compute_linear_allreduce(
        infinicore::Tensor &input, infinicclComm_t communicator) const;

    size_t in_features_;
    size_t out_features_;
    bool has_bias_;
    infinicore::DataType dtype_;
    int split_dim_ = -1;
    float alpha_ = 1.0f;
    std::string stem_; // checkpoint 张量名路径（只给按名字查表的量化方案用，见 §6.0 纠正 2）
    // init_fused_shards 记下的逐 shard checkpoint stem，下标 == parameters_ key 里的 i。
    // 与 key 在同一个循环里产生、forward 里消费，因此只是个局部不变量（不是跨阶段约定）；
    // 非融合路径为空。语义见 BaseQuantization::forward 的 shard_stems 重载。
    std::vector<std::string> shard_stems_;
    bool sharded_ = false; // 融合量化布局：本对象不持有融合 buffer，参数在 shard<i>.* 里
    std::shared_ptr<infinilm::quantization::BaseQuantization> quantization_;
};

} // namespace infinilm::nn
