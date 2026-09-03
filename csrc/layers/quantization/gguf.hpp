#pragma once

#include "base_quantization.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>

namespace infinilm::quantization {

// GGUF block quantization（路线 B）：打包器把 GGUF 张量的**原始块字节**逐字节搬进
// safetensors，一行的宽度是 row_bytes(in_features, type)，不是 in_features 个元素。
// 因此每个权重的布局只能由 checkpoint 张量名查表决定，逻辑形状推不出来。
//
// 类型表 = config.json:quantization_config.ggml_types，键就是 safetensors 里的张量名
// 原文（打包器自检保证与产物张量名双向逐字相等），值要么是 ggml type id，要么是
// 字符串 "dense_bf16"（打包期已反量化成 BF16 的那些：embed / lm_head / norm /
// GDN 标量 / v1 的 IQ4_*）。quantization_config.key_prefix 在这里裁掉一次，因为
// 挂在 model. 以下的模块不知道自己的绝对路径。详见执行方案 §2.3 / §6.0。
class GGUFBlockQuantization : public BaseQuantization {
public:
    // 稠密化条目在类型表里的取值（与任何 ggml type id 都不冲突：id 从 0 起）
    static constexpr int64_t DENSE_BF16 = -1;
    // blob 权重在 checkpoint 里的张量名后缀（与 scripts/gguf_mapping.BLOB_SUFFIX 一致）
    static constexpr const char *BLOB_SUFFIX = "weight_bytes";
    static constexpr const char *DENSE_SUFFIX = "weight";
    // 融合 Linear 在 parameters_ 里给各 shard 用的 key 前缀，见 BaseLinear::init_fused_shards
    static constexpr const char *SHARD_PREFIX = "shard";

    explicit GGUFBlockQuantization(const nlohmann::json &quant_config);

    ~GGUFBlockQuantization() override;

    QuantScheme get_quant_scheme() const override {
        return QuantScheme::GGUF_BLOCK;
    }

    // 名称未知的布局无法决定 ggml 类型，GGUF 只能通过带 stem 的重载被调用
    std::vector<ParamDescriptor> get_param_layout(
        size_t in_features, size_t out_features,
        int split_dim, int tp_rank, int tp_size,
        int tp_num_heads,
        const infinicore::DataType &dtype,
        bool bias) const override;

    std::vector<ParamDescriptor> get_param_layout(
        size_t in_features, size_t out_features,
        int split_dim, int tp_rank, int tp_size,
        int tp_num_heads,
        const infinicore::DataType &dtype,
        bool bias,
        const std::string &stem) const override;

    infinicore::Tensor forward(
        const ParamsMap &params,
        const infinicore::Tensor &input,
        bool has_bias,
        float alpha = 1.0f) const override;

    infinicore::Tensor forward(
        const ParamsMap &params,
        const infinicore::Tensor &input,
        bool has_bias,
        float alpha,
        const std::string &stem) const override;

    // 融合 Linear 的唯一入口：各 shard 的 ggml type id 只能由自己的 stem 查出来
    //（实测 q/k/v 同类型的 full-attn 层数 0/16），而组 stem 做不到。见 §7.2 子步骤 0。
    infinicore::Tensor forward(
        const ParamsMap &params,
        const infinicore::Tensor &input,
        bool has_bias,
        float alpha,
        const std::string &stem,
        const std::vector<std::string> &shard_stems) const override;

    // GGUF 的融合 Linear 不在 base_linear 里走这条路（各 shard 本来就是独立 buffer，
    // 没有可 narrow 的父 buffer），这里只做「shard<i> -> <prefix>.<suffix>」的名字映射，
    // 字节一个不动，供 BaseLinear::split_params 的既有调用点安全通过。
    std::vector<SplitParam> split_params(
        const std::unordered_map<std::string, infinicore::nn::Parameter> &params,
        const std::vector<SplitInfo> &splits,
        int narrow_dim,
        int tp_rank, int tp_size, int tp_num_heads) const override;

    // 不改写任何字节：blob 的语义就是「GGUF 原始字节」，一旦被 post-process
    // 加工就失去与 llama.cpp 逐 block 对拍的能力（方案 §4 的基准）。
    std::shared_ptr<BaseQuantization> process_weights_after_loading(
        ParamsMap &params,
        const infinicore::Device &device,
        int split_dim = -1) const override;

    // ---- 供自检 / 诊断使用 ----
    // stem -> ggml type id 或 DENSE_BF16。命中 0 个或 2 个候选都抛错：宁可拒启，
    // 也不能静默走稠密路径（能加载、显存暴涨、结果错）。
    // matched_key 非空时额外给出表里真正命中的那条键（已裁前缀的形态），报错里用它
    // 才能 grep 到；旧形态产物的 blob 键是归一成 `.weight` 的，不能拿 stem 拼凑。
    int64_t resolve(const std::string &stem, std::string *matched_key = nullptr) const;
    size_t row_bytes(size_t in_features, int64_t type_id) const;
    bool has_group(const std::string &group_stem) const;
    size_t table_size() const { return types_.size(); }

    static bool is_known_type(int64_t type_id);

private:
    // type_id = 本权重在类型表里的 ggml type id（稠密条目为 DENSE_BF16），阶段 3 的
    // kernel 分发靠它；table_key = 命中的表键（报错里给的名字必须能 grep 到）。
    infinicore::Tensor forward_shard(
        const std::string &suffix,
        const infinicore::Tensor &weight,
        const infinicore::Tensor &input,
        float alpha,
        int64_t type_id,
        const std::string &table_key) const;

    // 运行时激活 V 头置换（out_proj 一类「权重列需要重排」的条目）。
    // 为什么必须在运行时做：conversion/qwen.py:607-609 导出 GGUF 时把 ssm_out 的**列**
    // 从 grouped 换成了 tiled，而 GDN kernel 的 v 头序是 grouped（InfiniCore
    // chunk_gated_delta_rule/cuda/kernel.cuh:112 `key_head_idx = value_head_idx /
    // value_heads_per_key_head`）；blob 的块沿 in 维切（Q4_K/Q5_K/Q6_K block_size=256），
    // 打包期置换列 = 跨块重排 = 要重量化，做不到 ⇒ 只能把激活置换过去。
    // 规则不在这里硬编码，由打包器从映射表派生写进
    // config.json:quantization_config.activation_vperm（见 scripts/gguf_mapping.py）。
    struct ActVPerm {
        std::string suffix; // 尾匹配用，含结尾 '.'，例如 "linear_attn.out_proj."
        size_t n_k;         // key 头数
        size_t r;           // 每个 key 头带几个 value 头
        size_t hd;          // value head_dim
    };

    // stem 命中哪条规则（没有则 nullptr）。按后缀匹配，因为层号在 C++ 侧不可信。
    const ActVPerm *vperm_rule(const std::string &stem) const;

    // [..., n_k*r*hd]（grouped）-> [..., r*n_k*hd]（tiled），纯视图 + 一次 contiguous
    static infinicore::Tensor gather_grouped_to_tiled(
        const ActVPerm &rule, const infinicore::Tensor &input, const std::string &name);

    std::string describe(const std::string &stem) const;

    // 类型表条目。name = 它在 config.json:ggml_types 里的**原始键**（未裁前缀），
    // 只能靠它把报错写成可在产物里 grep 的名字：裁过前缀的键在旧形态产物里连后缀
    // 都不一样（blob 被归一成了 .weight），拿 stem 拼凑出来的名字两边都 grep 不到。
    struct TypeEntry {
        int64_t id;
        std::string name;
    };

    std::unordered_map<std::string, TypeEntry> types_;
    std::string key_prefix_;
    std::vector<ActVPerm> vperm_; // 见 ActVPerm（空 = config 声明本产物无需置换）
    // 命中统计（get_param_layout 是 const，所以 mutable）
    mutable size_t n_blob_ = 0;
    mutable size_t n_dense_ = 0;
    mutable size_t n_group_ = 0;
};

} // namespace infinilm::quantization
