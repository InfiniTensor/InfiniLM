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

// GGML block metadata. A packed row contains
// in_features / block_size * type_size bytes. Unsupported types must be
// converted to dense BF16 instead of relying on a guessed runtime stride.
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
            "GGUFBlockQuantization: quantization_config is missing ggml_types");
    }
    key_prefix_ = get_or<std::string>("key_prefix", "");

    const auto &table = quant_config_.at("ggml_types");
    if (!table.is_object() || table.empty()) {
        throw std::runtime_error("GGUFBlockQuantization: ggml_types is empty");
    }

    size_t n_blob = 0;
    size_t n_dense = 0;
    size_t n_outside = 0;
    for (const auto &kv : table.items()) {
        const std::string &name = kv.key();
        // Keep keys outside key_prefix unchanged. This includes root-level
        // tensors such as lm_head.weight and preserves exact checkpoint names.
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
                    "GGUFBlockQuantization: value '" + v + "' for '" + name
                    + "' is neither an integer type id nor \"" + DENSE_MARK + "\"");
            }
            ++n_dense;
        } else {
            if (!kv.value().is_number_integer()) {
                throw std::runtime_error(
                    "GGUFBlockQuantization: value for '" + name + "' is not an integer ggml type id");
            }
            id = kv.value().get<int64_t>();
            if (id == DENSE_BF16) {
                throw std::runtime_error(
                    "GGUFBlockQuantization: type id for '" + name + "' conflicts with dense sentinel -1");
            }
            if (!ggml_block(id)) {
                throw std::runtime_error(
                    "GGUFBlockQuantization: '" + name + "' uses unsupported ggml type id="
                    + std::to_string(id) + " (supported: " + supported_types()
                    + "); unsupported types must be converted to dense BF16");
            }
            ++n_blob;
        }

        if (!types_.emplace(std::move(key), TypeEntry{id, name}).second) {
            throw std::runtime_error(
                "GGUFBlockQuantization: duplicate key after removing key_prefix: '" + name + "'");
        }
    }

    // Activation value-head permutation is required even when the rule list is
    // empty. Missing metadata could silently misalign activations and columns.
    if (!quant_config_.contains("activation_vperm")) {
        throw std::runtime_error(
            "GGUFBlockQuantization: quantization_config is missing activation_vperm; "
            "refresh config.json with the converter --skip-pack option");
    }
    {
        const auto &rules = quant_config_.at("activation_vperm");
        if (!rules.is_array()) {
            throw std::runtime_error("GGUFBlockQuantization: activation_vperm must be an array, got " + std::string(rules.type_name()));
        }
        for (const auto &j : rules) {
            if (!j.is_object()) {
                throw std::runtime_error("GGUFBlockQuantization: activation_vperm entry must be an object");
            }
            ActVPerm r;
            for (const char *key : {"suffix", "num_k_heads", "num_v_per_k", "head_dim"}) {
                if (!j.contains(key)) {
                    throw std::runtime_error("GGUFBlockQuantization: activation_vperm entry is missing '" + std::string(key) + "'");
                }
            }
            r.suffix = j.at("suffix").get<std::string>();
            r.n_k = j.at("num_k_heads").get<size_t>();
            r.r = j.at("num_v_per_k").get<size_t>();
            r.hd = j.at("head_dim").get<size_t>();
            if (r.suffix.empty() || r.suffix.back() != '.' || !r.n_k || !r.r || !r.hd) {
                throw std::runtime_error(
                    "GGUFBlockQuantization: invalid activation_vperm entry: suffix='"
                    + r.suffix + "' must end with '.', and dimensions must be positive (got "
                    + std::to_string(r.n_k) + "/" + std::to_string(r.r) + "/"
                    + std::to_string(r.hd) + ")");
            }
            if (std::any_of(vperm_.begin(), vperm_.end(),
                            [&r](const ActVPerm &e) { return e.suffix == r.suffix; })) {
                throw std::runtime_error("GGUFBlockQuantization: duplicate activation_vperm suffix '" + r.suffix + "'");
            }
            vperm_.push_back(std::move(r));
        }
    }

    static std::atomic<bool> config_logged{false};
    if (!config_logged.exchange(true, std::memory_order_relaxed)) {
        spdlog::info(
            "GGUF block quantization: {} entries (blob {} / dense {} / outside prefix {}), key_prefix='{}'{}",
            types_.size(), n_blob, n_dense, n_outside, key_prefix_,
            key_prefix_.empty()
                ? " (not set; table keys are relative safetensors names)"
                : " (for example, root-level lm_head entries)");

        std::string vs;
        for (const auto &r : vperm_) {
            if (!vs.empty()) {
                vs += ", ";
            }
            vs += r.suffix + "=" + std::to_string(r.n_k) + "x" + std::to_string(r.r) + "x" + std::to_string(r.hd);
        }
        spdlog::info("GGUF block quantization: {} activation V-head permutation rules (grouped->tiled): {}",
                     vperm_.size(), vs.empty() ? "none" : vs);
    }
}

GGUFBlockQuantization::~GGUFBlockQuantization() {
    if (n_blob_ + n_dense_ + n_group_ > 0) {
        spdlog::debug("GGUF block quantization: layout matches blob {} / dense {} / fused group {}",
                      n_blob_, n_dense_, n_group_);
    }
}

bool GGUFBlockQuantization::is_known_type(int64_t type_id) {
    return ggml_block(type_id) != nullptr;
}

std::string GGUFBlockQuantization::describe(const std::string &stem) const {
    // Restore the absolute checkpoint name for searchable diagnostics.
    return (stem.empty() ? std::string("<empty stem>") : key_prefix_ + stem);
}

int64_t GGUFBlockQuantization::resolve(const std::string &stem, std::string *matched_key) const {
    const std::string blob_key = stem + BLOB_SUFFIX;
    const std::string dense_key = stem + DENSE_SUFFIX;
    const auto blob_it = types_.find(blob_key);
    const auto dense_it = types_.find(dense_key);
    const int hits = (blob_it != types_.end()) + (dense_it != types_.end());

    // Require exactly one packed or dense candidate. Falling back on missing
    // metadata could load successfully while producing incorrect output.
    if (hits != 1) {
        throw std::runtime_error(
            "GGUFBlockQuantization: stem '" + describe(stem) + "' matched "
            + std::to_string(hits) + " type-table candidates; expected exactly one of '"
            + blob_key + "' or '" + dense_key + "' among "
            + std::to_string(types_.size()) + " entries");
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
            "GGUFBlockQuantization: unsupported ggml type id=" + std::to_string(type_id)
            + " (supported: " + supported_types() + ")");
    }
    if (in_features % b->block_size != 0) {
        throw std::runtime_error(
            "GGUFBlockQuantization: in_features=" + std::to_string(in_features)
            + " is not divisible by " + b->name + " block size "
            + std::to_string(b->block_size));
    }
    return in_features / b->block_size * b->type_size;
}

const GGUFBlockQuantization::ActVPerm *GGUFBlockQuantization::vperm_rule(
    const std::string &stem) const {
    for (const auto &r : vperm_) {
        if (stem.size() >= r.suffix.size() && stem.compare(stem.size() - r.suffix.size(), r.suffix.size(), r.suffix) == 0) {
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
            "GGUFBlockQuantization: activation for " + name + " has rank="
            + std::to_string(ndim) + "; expected at least [..., in_features]");
    }
    const size_t K = shape[ndim - 1];
    const size_t want = rule.n_k * rule.r * rule.hd;
    if (K != want) {
        throw std::runtime_error(
            "GGUFBlockQuantization: activation last dimension for " + name + " is "
            + std::to_string(K) + ", expected num_k_heads*num_v_per_k*head_dim="
            + std::to_string(want) + "; head permutation cannot be applied to a shard");
    }
    // [..., n_k, r, hd] -> [..., r, n_k, hd], grouped to tiled order.
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
        "GGUFBlockQuantization: get_param_layout requires a checkpoint stem to resolve the ggml type");
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
            "GGUFBlockQuantization: missing checkpoint stem while constructing Linear (in="
            + std::to_string(in_features) + ", out=" + std::to_string(out_features) + ")");
    }
    if (tp_size != 1 || tp_rank != 0) {
        throw std::runtime_error(
            "GGUFBlockQuantization: tensor parallelism is not supported for packed GGUF weights: "
            + describe(stem));
    }
    if (bias) {
        throw std::runtime_error(
            "GGUFBlockQuantization: GGUF checkpoint has no bias tensor for " + describe(stem));
    }

    // A stem without a trailing '.' identifies a fused Linear group. Its
    // individual shard buffers are allocated by BaseLinear::init_fused_shards.
    if (stem.back() != '.') {
        if (!has_group(stem)) {
            throw std::runtime_error(
                "GGUFBlockQuantization: fused-group stem '" + stem
                + "' has no '" + stem + ".<shard>.*' entry in the type table");
        }
        ++n_group_;
        return {};
    }

    const int64_t id = resolve(stem);
    if (id == DENSE_BF16) {
        ++n_dense_;
        // The converter stored this tensor as dense BF16; use regular GEMM.
        return {{"weight", {out_features, in_features}, dtype, split_dim, tp_rank, tp_size}};
    }

    ++n_blob_;
    const size_t rb = row_bytes(in_features, id);
    return {{{BLOB_SUFFIX}, {out_features, rb}, infinicore::DataType::U8, split_dim, tp_rank, tp_size}};
}

infinicore::Tensor GGUFBlockQuantization::forward(
    const ParamsMap &, const infinicore::Tensor &, bool, float) const {
    throw std::runtime_error(
        "GGUFBlockQuantization: forward requires a checkpoint stem; fused Linear also requires shard_stems");
}

infinicore::Tensor GGUFBlockQuantization::forward_shard(
    const std::string &suffix,
    const infinicore::Tensor &weight,
    const infinicore::Tensor &input,
    float alpha,
    int64_t type_id,
    const std::string &table_key) const {
    if (suffix == DENSE_SUFFIX) {
        // The suffix selected by get_param_layout must agree with the type table.
        if (type_id != DENSE_BF16) {
            throw std::runtime_error(
                "GGUFBlockQuantization: " + table_key + " has parameter suffix "
                + DENSE_SUFFIX + " but type table reports ggml type id="
                + std::to_string(type_id));
        }
        auto x = input->is_contiguous() ? input : input->contiguous();
        auto w = weight->is_contiguous() ? weight : weight->contiguous();
        return infinicore::op::linear(x, w, std::nullopt, alpha);
    }
    if (suffix == BLOB_SUFFIX) {
        // Pass packed block bytes directly to the kernel. Never reinterpret
        // them as BF16 through a dense fallback.
        if (alpha != 1.0F) {
            throw std::runtime_error(
                "linear_gguf: alpha=" + std::to_string(alpha)
                + " is unsupported for packed GGUF weights: " + table_key);
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
        // linear_gguf selects the decode or prefill kernel using its shared
        // kMaxDecodeM threshold, so this layer does not duplicate that limit.

        auto flat = x->view({M, K});
        flat = flat->is_contiguous() ? flat : flat->contiguous();
        const bool f32_decode_out = use_f32_decode_output(table_key, M);
        const auto out_dtype = f32_decode_out
                                 ? infinicore::DataType::F32
                                 : input->dtype();
        auto out = infinicore::Tensor::empty({M, N}, out_dtype, input->device());
        // Log the first packed invocation as a lightweight wiring diagnostic.
        static std::atomic<long> blob_calls{0};
        if (blob_calls.fetch_add(1) == 0) {
            spdlog::info(
                "linear_gguf: first packed forward {} -- M={} N={} K={} ggml_type={} row_bytes={}",
                table_key, M, N, K, type_id, w->size(1));
        }
        if (f32_decode_out) {
            static std::atomic<long> f32_calls{0};
            if (f32_calls.fetch_add(1) == 0) {
                spdlog::warn(
                    "linear_gguf: experimental F32 decode output enabled; first match {} -- M={} N={} K={}",
                    table_key, M, N, K);
            }
        }
        infinicore::op::linear_gguf_(out, flat, w, type_id);

        std::vector<size_t> out_shape(x_shape.begin(), x_shape.end() - 1);
        out_shape.push_back(N);
        return out->view(out_shape);
    }
    throw std::runtime_error(
        "GGUFBlockQuantization: parameter suffix '" + suffix + "' for " + table_key
        + " is neither " + DENSE_SUFFIX + " nor " + BLOB_SUFFIX);
}

infinicore::Tensor GGUFBlockQuantization::forward(
    const ParamsMap &params,
    const infinicore::Tensor &input,
    bool has_bias,
    float alpha,
    const std::string &stem) const {
    // The overload below handles fused Linear layers with per-shard stems.
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
            "GGUFBlockQuantization: bias is not supported (" + describe(stem) + ")");
    }

    // Apply activation permutation before either packed or dense execution;
    // both layouts preserve the source GGUF column order.
    infinicore::Tensor x = input;
    const ActVPerm *rule = vperm_rule(stem);
    if (!shard_stems.empty()) {
        for (const auto &s : shard_stems) {
            if (vperm_rule(s)) {
                throw std::runtime_error(
                    "GGUFBlockQuantization: shard '" + describe(s) + "' in fused group '"
                    + describe(stem) + "' matches an activation-permutation rule, but one input "
                                       "cannot be permuted independently for each shard");
            }
        }
    } else if (rule) {
        x = gather_grouped_to_tiled(*rule, input, describe(stem));
        // Log the first permutation as a lightweight wiring diagnostic.
        static std::atomic<long> vperm_applied{0};
        if (vperm_applied.fetch_add(1) == 0) {
            spdlog::info(
                "linear_gguf: first activation V-head permutation {} -- grouped->tiled {}x{}x{}",
                describe(stem), rule->n_k, rule->r, rule->hd);
        }
    }

    // A non-fused layer owns exactly one weight or weight_bytes parameter.
    if (shard_stems.empty()) {
        if (params.size() != 1) {
            throw std::runtime_error(
                "GGUFBlockQuantization: " + describe(stem) + " has "
                + std::to_string(params.size())
                + " parameters but no shard_stems; BaseLinear::compute_linear did not pass them");
        }
        const auto &kv = *params.begin();
        std::string table_key;
        const int64_t id = resolve(stem, &table_key);
        return forward_shard(kv.first, kv.second, x, alpha, id, table_key);
    }

    // Fused parameters use shard<i>.<suffix>, where i is their order along the
    // output dimension and matches SplitInfo. Resolve each shard independently
    // and concatenate outputs in that order.
    if (shard_stems.size() != params.size()) {
        throw std::runtime_error(
            "GGUFBlockQuantization: " + describe(stem) + " has "
            + std::to_string(params.size()) + " shard parameters but received "
            + std::to_string(shard_stems.size())
            + " shard stems; both must be created by BaseLinear::init_fused_shards");
    }
    std::vector<std::pair<size_t, infinicore::Tensor>> parts;
    for (const auto &kv : params) {
        if (kv.first.compare(0, std::string(SHARD_PREFIX).size(), SHARD_PREFIX) != 0) {
            throw std::runtime_error(
                "GGUFBlockQuantization: fused Linear parameter '" + kv.first
                + "' does not match " + SHARD_PREFIX + "<i>.<suffix> (" + describe(stem) + ")");
        }
        const size_t dot = kv.first.find('.');
        if (dot == std::string::npos) {
            throw std::runtime_error(
                "GGUFBlockQuantization: fused Linear parameter '" + kv.first + "' is missing '.'");
        }
        const size_t idx = std::stoul(kv.first.substr(std::string(SHARD_PREFIX).size(),
                                                      dot - std::string(SHARD_PREFIX).size()));
        if (idx >= shard_stems.size()) {
            throw std::runtime_error(
                "GGUFBlockQuantization: shard index in parameter '" + kv.first
                + "' is outside shard_stems (" + describe(stem) + ")");
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
    // Fused GGUF shards already have independent buffers. Rename
    // shard<i>.<suffix> to <prefix>.<suffix> without slicing data.
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
            "GGUFBlockQuantization::split_params: expected "
            + std::to_string(splits.size()) + " shard parameters but matched "
            + std::to_string(result.size())
            + "; fused GGUF Linear must use BaseLinear::init_fused_shards");
    }
    return result;
}

std::shared_ptr<BaseQuantization> GGUFBlockQuantization::process_weights_after_loading(
    ParamsMap &params,
    const infinicore::Device &,
    int) const {
    for (auto &kv : params) {
        const bool is_blob = kv.first.size() >= strlen(BLOB_SUFFIX) && kv.first.compare(kv.first.size() - strlen(BLOB_SUFFIX), strlen(BLOB_SUFFIX), BLOB_SUFFIX) == 0;
        if (!is_blob) {
            continue;
        }
        if (kv.second->dtype() != infinicore::DataType::U8) {
            throw std::runtime_error(
                "GGUFBlockQuantization: packed parameter '" + kv.first + "' must have U8 dtype");
        }
        if (!kv.second->is_contiguous()) {
            throw std::runtime_error(
                "GGUFBlockQuantization: packed parameter '" + kv.first + "' must be contiguous");
        }
    }
    // Keep the quantization scheme and raw bytes unchanged.
    return nullptr;
}

} // namespace infinilm::quantization
