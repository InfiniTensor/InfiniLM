#include "gguf.hpp"

#include <infinicore/ops/cat.hpp>
#include <infinicore/ops/linear.hpp>
#include <infinicore/ops/linear_gguf.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace infinilm::quantization {

namespace {

// ggml 块的 (block_size, type_size)：一行 row_bytes = in / block_size * type_size。
// 数值取自 §2.3 的实测（与 gguf-py 的 GGML_QUANT_SIZES、llama.cpp 的 ggml.h 一致）。
// 只列本路线 kernel 计划支持的类型；表外的 id 一律抛错，逼着打包期把它稠密化，
// 而不是运行期猜一个 stride（猜错 = 读越界 = 结果错）。
struct GgmlBlock {
    int64_t id;
    const char *name;
    size_t block_size;
    size_t type_size;
};

constexpr GgmlBlock GGML_BLOCKS[] = {
    {8, "Q8_0", 32, 34},
    {12, "Q4_K", 256, 144},
    {13, "Q5_K", 256, 176},
    {14, "Q6_K", 256, 210},
};

const GgmlBlock *ggml_block(int64_t id) {
    for (const auto &b : GGML_BLOCKS) {
        if (b.id == id) {
            return &b;
        }
    }
    return nullptr;
}

std::string supported_types() {
    std::string s;
    for (const auto &b : GGML_BLOCKS) {
        if (!s.empty()) {
            s += "/";
        }
        s += b.name;
    }
    return s;
}

constexpr const char *DENSE_MARK = "dense_bf16";

bool env_enabled(const char *name) {
    const char *value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

bool use_f32_decode_output(const std::string &table_key, size_t m_count) {
    if (!env_enabled("INFINI_GGUF_F32_DECODE_OUT") || m_count > 16) {
        return false;
    }
    const char *match = std::getenv("INFINI_GGUF_F32_DECODE_OUT_MATCH");
    return match == nullptr || match[0] == '\0'
        || table_key.find(match) != std::string::npos;
}

} // namespace

GGUFBlockQuantization::GGUFBlockQuantization(const nlohmann::json &quant_config)
    : BaseQuantization(quant_config) {
    if (!quant_config_.is_object() || !quant_config_.contains("ggml_types")) {
        throw std::runtime_error(
            "GGUFBlockQuantization: quantization_config 缺 ggml_types（阶段 1 打包器写入）");
    }
    key_prefix_ = get_or<std::string>("key_prefix", "");

    const auto &table = quant_config_.at("ggml_types");
    if (!table.is_object() || table.empty()) {
        throw std::runtime_error("GGUFBlockQuantization: ggml_types 表为空");
    }

    size_t n_blob = 0;
    size_t n_dense = 0;
    size_t n_outside = 0;
    for (const auto &kv : table.items()) {
        const std::string &name = kv.key();
        // 实测：产物 121 张量里只有 lm_head.weight 不在 model.language_model. 子树下
        // （lm_head 在 C++ 模块树里是根节点的兄弟），这类键原样保留、不裁前缀。
        // 它们永远不会被 stem 查到（lm_head 走非量化 ctor），留着是为了让类型表
        // 与产物张量名保持双向逐字相等（阶段 1 自检的判据）。
        std::string key = name;
        if (!key_prefix_.empty() && name.compare(0, key_prefix_.size(), key_prefix_) == 0) {
            key = name.substr(key_prefix_.size());
        } else {
            ++n_outside;
        }

        int64_t id = DENSE_BF16;
        if (kv.value().is_string()) {
            const std::string v = kv.value().get<std::string>();
            if (v != DENSE_MARK) {
                throw std::runtime_error(
                    "GGUFBlockQuantization: '" + name + "' 的取值 '" + v + "' 既不是整数 type id 也不是 \"" +
                    DENSE_MARK + "\"");
            }
            ++n_dense;
        } else {
            if (!kv.value().is_number_integer()) {
                throw std::runtime_error(
                    "GGUFBlockQuantization: '" + name + "' 的取值不是整数 ggml type id");
            }
            id = kv.value().get<int64_t>();
            if (id == DENSE_BF16) {
                throw std::runtime_error(
                    "GGUFBlockQuantization: '" + name + "' 的 type id 与稠密标记 -1 冲突");
            }
            if (!ggml_block(id)) {
                throw std::runtime_error(
                    "GGUFBlockQuantization: '" + name + "' 是不支持的 ggml type id=" +
                    std::to_string(id) + "（当前支持 " + supported_types() +
                    "；其余类型必须在打包期稠密化，不能留到运行期猜）");
            }
            ++n_blob;
        }

        if (!types_.emplace(std::move(key), TypeEntry{id, name}).second) {
            throw std::runtime_error(
                "GGUFBlockQuantization: 裁掉 key_prefix 后键重复：'" + name + "'");
        }
    }

    // 激活 V 头置换规则（out_proj 一类「要置换的是权重列」的条目）。缺这个键 = 拒启，
    // 不静默不置换：漏一次置换 = 48 个 value head 与权重列整体错位
    // =「能加载、能跑、输出错」，正是阶段 4 §8.5 要排除的那一类错。
    if (!quant_config_.contains("activation_vperm")) {
        throw std::runtime_error(
            "GGUFBlockQuantization: quantization_config 缺 activation_vperm（out_proj 的 V 头"
            "列序置换规则）——旧产物用打包器 --skip-pack 刷新 config.json 即可，不必重打包权重");
    }
    {
        const auto &rules = quant_config_.at("activation_vperm");
        if (!rules.is_array()) {
            throw std::runtime_error("GGUFBlockQuantization: activation_vperm 必须是数组，实际是 " +
                                     std::string(rules.type_name()));
        }
        for (const auto &j : rules) {
            if (!j.is_object()) {
                throw std::runtime_error("GGUFBlockQuantization: activation_vperm 条目不是对象");
            }
            ActVPerm r;
            for (const char *key : {"suffix", "num_k_heads", "num_v_per_k", "head_dim"}) {
                if (!j.contains(key)) {
                    throw std::runtime_error("GGUFBlockQuantization: activation_vperm 条目缺 '" +
                                             std::string(key) + "'");
                }
            }
            r.suffix = j.at("suffix").get<std::string>();
            r.n_k = j.at("num_k_heads").get<size_t>();
            r.r = j.at("num_v_per_k").get<size_t>();
            r.hd = j.at("head_dim").get<size_t>();
            if (r.suffix.empty() || r.suffix.back() != '.' || !r.n_k || !r.r || !r.hd) {
                throw std::runtime_error(
                    "GGUFBlockQuantization: activation_vperm 条目不合法：suffix='" + r.suffix +
                    "' 需以 '.' 结尾，三个维度需为正（实际 " + std::to_string(r.n_k) + "/" +
                    std::to_string(r.r) + "/" + std::to_string(r.hd) + "）");
            }
            if (std::any_of(vperm_.begin(), vperm_.end(),
                            [&r](const ActVPerm &e) { return e.suffix == r.suffix; })) {
                throw std::runtime_error("GGUFBlockQuantization: activation_vperm 里 '" + r.suffix +
                                         "' 出现多次（同一条规则只能有一份）");
            }
            vperm_.push_back(std::move(r));
        }
    }

    // n_outside 有两种成因，得分开说：早先全量产物未声明 key_prefix，整张表都被计入
    // 「前缀外」（实测日志里印成「前缀外 947」），很容易被读成「947 条都查不到」。
    spdlog::info(
        "GGUF block quantization: 类型表 {} 条（blob {} / 稠密 {} / 未裁前缀 {}），key_prefix='{}'{}",
        types_.size(), n_blob, n_dense, n_outside, key_prefix_,
        key_prefix_.empty()
            ? "（未声明：表键即 safetensors 张量名的相对形态，整表不裁前缀）"
            : "（在 prefix 之外，如 lm_head）");

    // 与下一行一起构成「本次加载到底有没有在做置换」的唯一可 grep 证据（A/B 靠它）
    std::string vs;
    for (const auto &r : vperm_) {
        if (!vs.empty()) {
            vs += ", ";
        }
        vs += r.suffix + "=" + std::to_string(r.n_k) + "x" + std::to_string(r.r) + "x" +
              std::to_string(r.hd);
    }
    spdlog::info("GGUF block quantization: 激活 V 头置换规则 {} 条（grouped->tiled）：{}",
                 vperm_.size(), vs.empty() ? "无" : vs);
}

GGUFBlockQuantization::~GGUFBlockQuantization() {
    if (n_blob_ + n_dense_ + n_group_ > 0) {
        spdlog::info("GGUF block quantization: 布局查表命中 blob {} / 稠密 {} / 融合组 {}",
                     n_blob_, n_dense_, n_group_);
    }
}

bool GGUFBlockQuantization::is_known_type(int64_t type_id) {
    return ggml_block(type_id) != nullptr;
}

std::string GGUFBlockQuantization::describe(const std::string &stem) const {
    // 报错信息里拼回绝对名，方便直接在产物 / pack_report.json 里 grep
    return (stem.empty() ? std::string("<空 stem>") : key_prefix_ + stem);
}

int64_t GGUFBlockQuantization::resolve(const std::string &stem, std::string *matched_key) const {
    const std::string blob_key = stem + BLOB_SUFFIX;
    const std::string dense_key = stem + DENSE_SUFFIX;
    const auto blob_it = types_.find(blob_key);
    const auto dense_it = types_.find(dense_key);
    const int hits = (blob_it != types_.end()) + (dense_it != types_.end());

    // 命中 0 个 = 拼错或产物缺张量；命中 2 个 = 打包器同时写了 blob 与稠密版本。
    // 两种都必须是异常：任何「查不到就走稠密」的回落都会变成能加载、显存暴涨、结果错。
    if (hits != 1) {
        throw std::runtime_error(
            "GGUFBlockQuantization: stem '" + describe(stem) + "' 在类型表里命中 " +
            std::to_string(hits) + " 个候选（期望恰好 1 个：'" + blob_key + "' 或 '" + dense_key +
            "'）；表共 " + std::to_string(types_.size()) + " 条");
    }
    const auto &hit = blob_it != types_.end() ? *blob_it : *dense_it;
    if (matched_key) {
        *matched_key = hit.second.name;
    }
    return hit.second.id;
}

bool GGUFBlockQuantization::has_group(const std::string &group_stem) const {
    const std::string head = group_stem + ".";
    return std::any_of(types_.begin(), types_.end(), [&head](const auto &kv) {
        return kv.first.compare(0, head.size(), head) == 0;
    });
}

size_t GGUFBlockQuantization::row_bytes(size_t in_features, int64_t type_id) const {
    const GgmlBlock *b = ggml_block(type_id);
    if (!b) {
        throw std::runtime_error(
            "GGUFBlockQuantization: 不支持的 ggml type id=" + std::to_string(type_id) +
            "（当前支持 " + supported_types() + "）");
    }
    if (in_features % b->block_size != 0) {
        throw std::runtime_error(
            "GGUFBlockQuantization: in_features=" + std::to_string(in_features) +
            " 不能被 " + b->name + " 的块大小 " + std::to_string(b->block_size) + " 整除");
    }
    return in_features / b->block_size * b->type_size;
}

const GGUFBlockQuantization::ActVPerm *GGUFBlockQuantization::vperm_rule(
    const std::string &stem) const {
    for (const auto &r : vperm_) {
        if (stem.size() >= r.suffix.size() &&
            stem.compare(stem.size() - r.suffix.size(), r.suffix.size(), r.suffix) == 0) {
            return &r;
        }
    }
    return nullptr;
}

infinicore::Tensor GGUFBlockQuantization::gather_grouped_to_tiled(
    const ActVPerm &rule, const infinicore::Tensor &input, const std::string &name) {
    const auto shape = input->shape();
    const size_t ndim = shape.size();
    if (ndim < 2) {
        throw std::runtime_error(
            "GGUFBlockQuantization: " + name + " 的激活 rank=" + std::to_string(ndim) +
            "，至少要是 [..., in_features]");
    }
    const size_t K = shape[ndim - 1];
    const size_t want = rule.n_k * rule.r * rule.hd;
    if (K != want) {
        // TP 会把 in 维切成没关头数不等的分片，套上整头置换就是静默错位；
        // 与 get_param_layout 里「暂不支持 tensor parallel」的护栏保持同一口径。
        throw std::runtime_error(
            "GGUFBlockQuantization: " + name + " 的激活末维 " + std::to_string(K) +
            " != activation_vperm 的 num_k_heads*num_v_per_k*head_dim = " + std::to_string(want) +
            "（切分后的分片不能套整头置换）");
    }
    // [..., n_k, r, hd] -> [..., r, n_k, hd]：把 grouped（k-major）的激活置换为 tiled（v-major）。
    const size_t k_axis = ndim - 1;
    infinicore::Shape grouped(shape.begin(), shape.end() - 1);
    grouped.insert(grouped.end(), {rule.n_k, rule.r, rule.hd});
    infinicore::Shape order;
    order.reserve(grouped.size());
    for (size_t a = 0; a + 1 < ndim; ++a) {
        order.push_back(a);
    }
    order.insert(order.end(), {k_axis + 1, k_axis, k_axis + 2});

    auto x = input->is_contiguous() ? input : input->contiguous();
    return x->view(grouped)->permute(order)->contiguous()->view(shape);
}

std::vector<ParamDescriptor> GGUFBlockQuantization::get_param_layout(
    size_t, size_t, int, int, int, int,
    const infinicore::DataType &, bool) const {
    throw std::runtime_error(
        "GGUFBlockQuantization: 不接受无名字的 get_param_layout 调用（每个权重的 ggml "
        "类型只能由 checkpoint 张量名决定）");
}

std::vector<ParamDescriptor> GGUFBlockQuantization::get_param_layout(
    size_t in_features, size_t out_features,
    int split_dim, int tp_rank, int tp_size,
    int tp_num_heads,
    const infinicore::DataType &dtype,
    bool bias,
    const std::string &stem) const {
    (void)tp_num_heads;

    if (stem.empty()) {
        throw std::runtime_error(
            "GGUFBlockQuantization: 构造 Linear 时没有传 checkpoint stem（in=" +
            std::to_string(in_features) + ", out=" + std::to_string(out_features) +
            "）——方案 §6.1 列出的构造点必须全部补上 prefix/stem");
    }
    if (tp_size != 1 || tp_rank != 0) {
        throw std::runtime_error(
            "GGUFBlockQuantization: 暂不支持 tensor parallel（blob 的 TP 切分留待多卡阶段）：" +
            describe(stem));
    }
    if (bias) {
        throw std::runtime_error(
            "GGUFBlockQuantization: GGUF 产物里没有 bias 张量：" + describe(stem));
    }

    // 不带结尾 '.' 的 stem 表示「融合 Linear」：本类不为它分配任何 buffer，
    // 各 shard 由 BaseLinear::init_fused_shards 用各自的 stem 单独申请。
    if (stem.back() != '.') {
        if (!has_group(stem)) {
            throw std::runtime_error(
                "GGUFBlockQuantization: 融合组 stem '" + stem +
                "' 在类型表里没有任何 '" + stem + ".<shard>.*' 条目");
        }
        ++n_group_;
        return {};
    }

    const int64_t id = resolve(stem);
    if (id == DENSE_BF16) {
        ++n_dense_;
        // 与 NoneQuantization 同形：打包期已反量化成 BF16，正常 GEMM
        return {{"weight", {out_features, in_features}, dtype, split_dim, tp_rank, tp_size}};
    }

    ++n_blob_;
    const size_t rb = row_bytes(in_features, id);
    return {{{BLOB_SUFFIX}, {out_features, rb}, infinicore::DataType::U8, split_dim, tp_rank, tp_size}};
}

infinicore::Tensor GGUFBlockQuantization::forward(
    const ParamsMap &, const infinicore::Tensor &, bool, float) const {
    throw std::runtime_error(
        "GGUFBlockQuantization: 不接受无名字的 forward 调用（每个权重的 ggml 类型只能由 "
        "checkpoint 名字决定，融合 Linear 还需要 shard_stems）");
}

infinicore::Tensor GGUFBlockQuantization::forward_shard(
    const std::string &suffix,
    const infinicore::Tensor &weight,
    const infinicore::Tensor &input,
    float alpha,
    int64_t type_id,
    const std::string &table_key) const {
    if (suffix == DENSE_SUFFIX) {
        // 参数后缀是 get_param_layout 按 resolve() 结果选的，两者不一致 = 有地方改坏了
        //（blob 被当成 BF16 读就是「能加载、结果错」），宁可抛。
        if (type_id != DENSE_BF16) {
            throw std::runtime_error(
                "GGUFBlockQuantization: " + table_key + " 的参数后缀是 " + DENSE_SUFFIX +
                "，但类型表给出的 ggml type id=" + std::to_string(type_id) + "（不一致）");
        }
        auto x = input->is_contiguous() ? input : input->contiguous();
        auto w = weight->is_contiguous() ? weight : weight->contiguous();
        return infinicore::op::linear(x, w, std::nullopt, alpha);
    }
    if (suffix == BLOB_SUFFIX) {
        // 权重保持量化形态：块字节直接喂 kernel。这里绝不静默回落稠密 GEMM——
        // 那等于把块字节当成 BF16 读，能跑完但结果是错的，宁可抛。
        if (alpha != 1.0F) {
            throw std::runtime_error(
                "linear_gguf: 不支持 alpha=" + std::to_string(alpha) +
                "（GGUF blob 路径没有缩放权重，alpha!=1 说明上层期望与实现不符）：" +
                table_key);
        }
        auto x = input->is_contiguous() ? input : input->contiguous();
        auto w = weight->is_contiguous() ? weight : weight->contiguous();

        const auto x_shape = x->shape();
        const size_t ndim = x_shape.size();
        const size_t K = x_shape[ndim - 1];
        size_t M = 1;
        for (size_t i = 0; i + 1 < ndim; ++i) {
            M *= x_shape[i];
        }
        const size_t N = static_cast<size_t>(w->size(0));
        // 这里不再设 M 上限：gemv（小 M）与 prefill（大 M）两条路径在
        // linear_gguf 算子内部按同一个 kMaxDecodeM 谓词选。上层再留一份数字，
        // 两边一旦不同步就只剩一条过时的门（阶段 3.3 之前正是这种情形）。

        auto flat = x->view({M, K});
        flat = flat->is_contiguous() ? flat : flat->contiguous();
        const bool f32_decode_out = use_f32_decode_output(table_key, M);
        const auto out_dtype = f32_decode_out
            ? infinicore::DataType::F32
            : input->dtype();
        auto out = infinicore::Tensor::empty({M, N}, out_dtype, input->device());
        // 只报第一个 blob 调用：端到端排障时区分「死在 blob 路径之前」与
        //「已在 kernel 里」，两者处置完全不同（前者是接线问题，后者是下游算子）。
        static std::atomic<long> blob_calls{0};
        if (blob_calls.fetch_add(1) == 0) {
            spdlog::info(
                "linear_gguf: 首个 blob 前向 {} — M={} N={} K={} ggml_type={} row_bytes={}",
                table_key, M, N, K, type_id, w->size(1));
        }
        if (f32_decode_out) {
            static std::atomic<long> f32_calls{0};
            if (f32_calls.fetch_add(1) == 0) {
                spdlog::warn(
                    "linear_gguf: 实验性 F32 decode 输出已启用，首个命中 {} — M={} N={} K={}",
                    table_key, M, N, K);
            }
        }
        infinicore::op::linear_gguf_(out, flat, w, type_id);

        std::vector<size_t> out_shape(x_shape.begin(), x_shape.end() - 1);
        out_shape.push_back(N);
        return out->view(out_shape);
    }
    throw std::runtime_error(
        "GGUFBlockQuantization: " + table_key + " 的参数后缀 '" + suffix +
        "' 既不是 " + DENSE_SUFFIX + " 也不是 " + BLOB_SUFFIX);
}

infinicore::Tensor GGUFBlockQuantization::forward(
    const ParamsMap &params,
    const infinicore::Tensor &input,
    bool has_bias,
    float alpha,
    const std::string &stem) const {
    // 没有 shard stems 就只能服务非融合布局；融合 Linear 走下面那个重载。
    return forward(params, input, has_bias, alpha, stem, {});
}

infinicore::Tensor GGUFBlockQuantization::forward(
    const ParamsMap &params,
    const infinicore::Tensor &input,
    bool has_bias,
    float alpha,
    const std::string &stem,
    const std::vector<std::string> &shard_stems) const {
    if (has_bias) {
        throw std::runtime_error(
            "GGUFBlockQuantization: 不支持 bias（" + describe(stem) + "）");
    }

    // 先按规则置换激活，再进 blob / 稠密两条路：两条路的权重列序都直接来自同一个 GGUF
    // 张量（稠密化只换 dtype 不动列序），需要置换的语义完全一致。
    infinicore::Tensor x = input;
    const ActVPerm *rule = vperm_rule(stem);
    if (!shard_stems.empty()) {
        for (const auto &s : shard_stems) {
            if (vperm_rule(s)) {
                throw std::runtime_error(
                    "GGUFBlockQuantization: 融合组 '" + describe(stem) + "' 的 shard '"
                    + describe(s) + "' 命中激活置换规则，但一根 input 同时服务于所有 shard，"
                    "无法按 shard 分别置换（实际产物里 out_proj 不是融合 Linear，走到这里=接线错）");
            }
        }
    } else if (rule) {
        x = gather_grouped_to_tiled(*rule, input, describe(stem));
        // 只报第一次：端到端排障时它是「gather 真的在跑」的唯一证据，不靠日志量堆
        static std::atomic<long> vperm_applied{0};
        if (vperm_applied.fetch_add(1) == 0) {
            spdlog::info(
                "linear_gguf: 首个激活 V 头置换 {} — grouped->tiled {}x{}x{}",
                describe(stem), rule->n_k, rule->r, rule->hd);
        }
    }

    // 非融合：一个参数（weight 或 weight_bytes），stem 就是它自己的完整 checkpoint 路径
    if (shard_stems.empty()) {
        if (params.size() != 1) {
            throw std::runtime_error(
                "GGUFBlockQuantization: " + describe(stem) + " 有 " +
                std::to_string(params.size()) + " 个参数却没收到 shard_stems"
                "（内部错误：BaseLinear::compute_linear 没有把 shard_stems_ 传下来）");
        }
        const auto &kv = *params.begin();
        std::string table_key;
        const int64_t id = resolve(stem, &table_key);
        return forward_shard(kv.first, kv.second, x, alpha, id, table_key);
    }

    // 融合：parameters_ 里是 shard<i>.<suffix>，i 就是它们在输出 dim(-1) 上的顺序，
    // 与融合 Linear 的 SplitInfo 顺序一致 —— 所以输出拼回一根连续的 [.., sum(out_i)]，
    // 上层的 narrow 逻辑完全不用改（方案 §6.0 纠正 1）。
    // 每个 shard 的 ggml 类型由 shard_stems[i] 查表（实测 q/k/v 不同类型，见 §7.2）。
    if (shard_stems.size() != params.size()) {
        throw std::runtime_error(
            "GGUFBlockQuantization: " + describe(stem) + " 有 " +
            std::to_string(params.size()) + " 个 shard 参数但收到 " +
            std::to_string(shard_stems.size()) + " 个 shard stem（内部错误：两者应在 "
            "BaseLinear::init_fused_shards 的同一个循环里产生）");
    }
    std::vector<std::pair<size_t, infinicore::Tensor>> parts;
    for (const auto &kv : params) {
        if (kv.first.compare(0, std::string(SHARD_PREFIX).size(), SHARD_PREFIX) != 0) {
            throw std::runtime_error(
                "GGUFBlockQuantization: 融合 Linear 的参数名 '" + kv.first +
                "' 不是 " + SHARD_PREFIX + "<i>.<suffix> 形式（" + describe(stem) + "）");
        }
        const size_t dot = kv.first.find('.');
        if (dot == std::string::npos) {
            throw std::runtime_error(
                "GGUFBlockQuantization: 融合 Linear 的参数名 '" + kv.first + "' 缺 '.'");
        }
        const size_t idx = std::stoul(kv.first.substr(std::string(SHARD_PREFIX).size(),
                                                      dot - std::string(SHARD_PREFIX).size()));
        if (idx >= shard_stems.size()) {
            throw std::runtime_error(
                "GGUFBlockQuantization: 参数名 '" + kv.first + "' 的 shard 下标越出 shard_stems（" +
                describe(stem) + "）");
        }
        std::string table_key;
        const int64_t id = resolve(shard_stems[idx], &table_key);
        parts.emplace_back(idx, forward_shard(kv.first.substr(dot + 1), kv.second, x, alpha,
                                              id, table_key));
    }
    std::sort(parts.begin(), parts.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });

    std::vector<infinicore::Tensor> outs;
    outs.reserve(parts.size());
    for (auto &p : parts) {
        outs.push_back(p.second);
    }
    const auto shape = input->shape();
    return infinicore::op::cat(outs, static_cast<int>(shape.size()) - 1);
}

std::vector<SplitParam> GGUFBlockQuantization::split_params(
    const std::unordered_map<std::string, infinicore::nn::Parameter> &params,
    const std::vector<SplitInfo> &splits,
    int, int, int, int) const {
    // 恒等映射：GGUF 的融合 Linear 已经按 shard 分配了独立 buffer（没有可 narrow 的父
    // buffer），这里只把 shard<i>.<suffix> 换成 <prefix>.<suffix> 交给 register_fn。
    std::vector<SplitParam> result;
    for (size_t i = 0; i < splits.size(); ++i) {
        const std::string head = std::string(SHARD_PREFIX) + std::to_string(i) + ".";
        for (const auto &kv : params) {
            if (kv.first.compare(0, head.size(), head) != 0) {
                continue;
            }
            result.push_back({splits[i].prefix + "." + kv.first.substr(head.size()),
                              infinicore::nn::Parameter(kv.second)});
        }
    }
    if (result.size() != splits.size()) {
        throw std::runtime_error(
            "GGUFBlockQuantization::split_params: " + std::to_string(splits.size()) +
            " 个 shard 只匹配到 " + std::to_string(result.size()) +
            " 个参数（GGUF 融合 Linear 应走 BaseLinear::init_fused_shards）");
    }
    return result;
}

std::shared_ptr<BaseQuantization> GGUFBlockQuantization::process_weights_after_loading(
    ParamsMap &params,
    const infinicore::Device &,
    int) const {
    for (auto &kv : params) {
        const bool is_blob = kv.first.size() >= strlen(BLOB_SUFFIX) &&
                             kv.first.compare(kv.first.size() - strlen(BLOB_SUFFIX),
                                              strlen(BLOB_SUFFIX), BLOB_SUFFIX) == 0;
        if (!is_blob) {
            continue;
        }
        if (kv.second->dtype() != infinicore::DataType::U8) {
            throw std::runtime_error(
                "GGUFBlockQuantization: blob 参数 '" + kv.first + "' 的 dtype 不是 U8");
        }
        if (!kv.second->is_contiguous()) {
            throw std::runtime_error(
                "GGUFBlockQuantization: blob 参数 '" + kv.first + "' 不连续（阶段 3 kernel 按行取字节）");
        }
    }
    // 返回 nullptr：不换方案、不改写字节
    return nullptr;
}

} // namespace infinilm::quantization
