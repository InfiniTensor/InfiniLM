#include "../../csrc/config/inference_memory.hpp"
#include "../../csrc/models/llama/llama_for_causal_lm.hpp"
#include "../../csrc/weights/weights.hpp"
#ifdef NDEBUG
#undef NDEBUG
#endif

#include "../../csrc/global_state/global_state.hpp"
#include "../../csrc/models/minicpm4/minicpm4_for_causal_lm.hpp"
#include "runtime/context.hpp"
#include "runtime/runtime.hpp"

#include "../../csrc/layers/linear/base_linear.hpp"
#include <cassert>
#include <cstdio>
#include <exception>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
using namespace infinilm::models::minicpm4;
using ModelConfigPtr = std::shared_ptr<infinilm::config::ModelConfig>;
using Device = infinilm::runtime::Device;

class CountedAttention final : public MiniCPM4Attention {
public:
    using MiniCPM4Attention::MiniCPM4Attention;
    void process_weights_after_loading(infinilm::runtime::Runtime &runtime, bool pre_transpose) override {
        ++loads;
        MiniCPM4Attention::process_weights_after_loading(runtime, pre_transpose);
    }
    int loads = 0;
};

class CountedMLP final : public MiniCPM4MLP {
public:
    using MiniCPM4MLP::MiniCPM4MLP;
    void process_weights_after_loading(infinilm::runtime::Runtime &runtime, bool pre_transpose) override {
        ++loads;
        MiniCPM4MLP::process_weights_after_loading(runtime, pre_transpose);
    }
    int loads = 0;
};

class DecoderProbe final : public MiniCPM4DecoderLayer {
public:
    DecoderProbe(infinilm::runtime::Runtime &runtime, ModelConfigPtr config,
                 const infinilm::engine::distributed::RankInfo &rank,
                 infinilm::backends::AttentionBackend backend,
                 const infinilm::layers::moe::EPConfig &ep)
        : MiniCPM4DecoderLayer(runtime, config, 0, rank, backend, ep) {
        attention = bind_child("self_attn", std::make_shared<CountedAttention>(runtime, config, 0, rank, backend, ep));
        mlp = bind_child("mlp", std::make_shared<CountedMLP>(runtime, config, 0, rank, backend, ep));
        self_attn_ = attention;
        mlp_ = mlp;
    }
    std::shared_ptr<CountedAttention> attention;
    std::shared_ptr<CountedMLP> mlp;
};

class ModelProbe final : public infinilm::InfinilmModel {
public:
    ModelProbe(infinilm::runtime::Runtime &runtime, ModelConfigPtr config,
               const infinilm::engine::distributed::RankInfo &rank,
               infinilm::backends::AttentionBackend backend,
               const infinilm::layers::moe::EPConfig &ep) {
        decoder = bind_child("decoder", std::make_shared<DecoderProbe>(runtime, config, rank, backend, ep));
    }
    Output forward(const Input &, infinilm::global_state::ForwardContext &) const override {
        throw std::logic_error("This probe only tests production post-load traversal");
    }
    std::shared_ptr<DecoderProbe> decoder;
};

// No Tensor or Runtime is created: test the root virtual boundary itself.
class ForwardContextProbe final : public infinilm::InfinilmModel {
public:
    Output forward(const Input &, infinilm::global_state::ForwardContext &context) const override {
        observed = &context;
        observed_bound = context.attn_metadata.logical_kv_upper_bound;
        return {};
    }
    mutable infinilm::global_state::ForwardContext *observed{nullptr};
    mutable size_t observed_bound{0};
};

void check(bool transpose, infinilm::runtime::Runtime &runtime,
           const std::shared_ptr<infinilm::global_state::InfinilmConfig> &settings,
           const infinilm::engine::distributed::RankInfo &rank) {
    auto config = std::make_shared<infinilm::config::ModelConfig>(nlohmann::json{
        {"model_type", "minicpm4"}, {"dtype", "float32"}, {"hidden_size", 32}, {"vocab_size", 64}, {"head_dim", 16}, {"intermediate_size", 48}, {"num_attention_heads", 2}, {"num_key_value_heads", 1}, {"num_hidden_layers", 1}, {"max_position_embeddings", 64}, {"rms_norm_eps", 1e-5}, {"rope_theta", 10000.0}, {"attention_bias", false}, {"mlp_bias", false}});
    settings->model_config = config;
    // The explicit post-load argument, not the opposite configuration value, selects the layout.
    settings->pre_transpose = !transpose;
    ModelProbe model{runtime, config, rank, settings->attention_backend, infinilm::layers::moe::EPConfig{}};

    struct Weight {
        infinilm::runtime::Shape shape;
        std::vector<float> values;
    };
    std::map<std::string, Weight> originals;
    for (auto [name, tensor] : model.state_dict()) {
        auto contiguous = tensor->contiguous(runtime);
        assert(contiguous->data() == tensor->data());
        std::vector<float> values(tensor->numel());
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = static_cast<float>(static_cast<int>(i % 19) - 9) / 32.0F;
        }
        runtime.memcpyH2D(tensor->data(), values.data(), values.size() * sizeof(float), false);
        originals.emplace(name, Weight{tensor->shape(), std::move(values)});
    }
    model.process_weights_after_loading(runtime, transpose);
    const auto &attention = *model.decoder->attention;
    const auto &mlp = *model.decoder->mlp;
    std::printf("transpose=%d attention loads=%d mlp loads=%d\n",
                transpose, attention.loads, mlp.loads);
    if (attention.loads != 1 || mlp.loads != 1) {
        throw std::runtime_error("registered children were processed more than once");
    }
    const auto result = model.state_dict();
    assert(result.size() == originals.size());
    for (const auto &[name, original] : originals) {
        const auto &tensor = result.at(name);
        auto expected_shape = original.shape;
        std::vector<float> expected = original.values;
        if (transpose && expected_shape.size() == 2) {
            const size_t rows = expected_shape[0], cols = expected_shape[1];
            expected_shape = {cols, rows};
            for (size_t row = 0; row < rows; ++row) {
                for (size_t col = 0; col < cols; ++col) {
                    expected[col * rows + row] = original.values[row * cols + col];
                }
            }
        }
        assert(tensor->shape() == expected_shape);
        auto contiguous = tensor->contiguous(runtime);
        std::vector<float> actual(expected.size());
        runtime.memcpyD2H(actual.data(), contiguous->data(), actual.size() * sizeof(float));
        assert(actual == expected);
    }
}
// Exercise supplied-Runtime post-load traversal without a Context Runtime binding.
struct FirstQuantError final : std::runtime_error {
    FirstQuantError() : std::runtime_error("typed-postload-first-quant-error") {}
};

class CountedNone final : public infinilm::quantization::NoneQuantization {
public:
    explicit CountedNone(bool fail = false) : fail_(fail) {}
    std::shared_ptr<infinilm::quantization::BaseQuantization> process_weights_after_loading(
        infinilm::runtime::Runtime &runtime, infinilm::quantization::ParamsMap &params,
        bool transpose, int split_dim) const override {
        ++typed_calls;
        if (fail_) {
            throw FirstQuantError();
        }
        return NoneQuantization::process_weights_after_loading(runtime, params, transpose, split_dim);
    }
    mutable size_t typed_calls = 0;

private:
    bool fail_;
};

void typed_require(bool ok, const char *message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

void check_forward_context_dispatch() {
    ForwardContextProbe probe;
    const infinilm::InfinilmModel &root = probe;
    infinilm::InfinilmModel::Input input;
    infinilm::global_state::ForwardContext first, second;
    first.attn_metadata.logical_kv_upper_bound = 17;
    second.attn_metadata.logical_kv_upper_bound = 31;
    (void)root.forward(input, first);
    typed_require(probe.observed == &first && probe.observed_bound == 17,
                  "root forward did not propagate first explicit ForwardContext");
    (void)root.forward(input, second);
    typed_require(probe.observed == &second && probe.observed_bound == 31,
                  "root forward reused an implicit or previous ForwardContext");
    typed_require(first.runtime == nullptr && second.runtime == nullptr,
                  "root dispatch created an implicit Runtime");
    std::puts("root forward explicit two-context no-Tensor dispatch PASS");
}

std::vector<float> typed_read(infinilm::runtime::Runtime &runtime, infinilm::runtime::Tensor value) {
    auto contiguous = value->contiguous(runtime);
    std::vector<float> result(value->numel());
    runtime.memcpyD2H(result.data(), contiguous->data(), result.size() * sizeof(float));
    return result;
}

void typed_load(infinilm::runtime::Runtime &runtime, infinilm::nn::BaseLinear &linear) {
    // Frozen independent non-square weight [out=3,in=2], bias and input oracle.
    const float weights[]{1, 2, 3, 4, 5, 6};
    const float bias[]{7, 8, 9};
    runtime.memcpyH2D(linear.weight()->data(), weights, sizeof(weights), false);
    runtime.memcpyH2D(linear.bias()->data(), bias, sizeof(bias), false);
}

template <class Fn>
void typed_rejected(Fn &&fn) {
    bool rejected = false;
    try {
        fn();
    } catch (const std::exception &) { rejected = true; }
    typed_require(rejected, "typed operation unexpectedly accepted");
}

void typed_success_case(infinilm::runtime::Runtime &runtime, bool first_mode, bool repeat_mode) {
    auto quant = std::make_shared<CountedNone>();
    infinilm::nn::BaseLinear linear(runtime, 2, 3, quant, true, infinilm::runtime::DataType::kFloat32);
    typed_load(runtime, linear);
    infinilm::weights::ModelWeights *module = &linear; // Real virtual dispatch, no concrete overload call.
    module->process_weights_after_loading(runtime, first_mode);
    runtime.syncStream();
    const std::vector<float> expected_weight = first_mode
                                                 ? std::vector<float>{1, 3, 5, 2, 4, 6}
                                                 : std::vector<float>{1, 2, 3, 4, 5, 6};
    typed_require(linear.weight()->shape() == (first_mode ? infinilm::runtime::Shape{2, 3} : infinilm::runtime::Shape{3, 2}), "typed NONE shape");
    typed_require(typed_read(runtime, linear.weight()) == expected_weight, "typed NONE raw markers");
    typed_require(typed_read(runtime, linear.bias()) == std::vector<float>{7, 8, 9}, "typed NONE bias changed");
    const float inputs[]{2, 3};
    auto input = infinilm::runtime::Tensor::empty(runtime, {1, 1, 2}, infinilm::runtime::DataType::kFloat32);
    runtime.memcpyH2D(input->data(), inputs, sizeof(inputs), false);
    auto output = linear.forward(runtime, input);
    typed_require(output->shape() == infinilm::runtime::Shape{1, 1, 3}, "typed NONE forward shape");
    typed_require(typed_read(runtime, output) == std::vector<float>{15, 26, 37}, "typed NONE independent forward oracle");
    auto published = linear.weight();
    typed_rejected([&] { module->process_weights_after_loading(runtime, repeat_mode); });
    typed_require(quant->typed_calls == 1, "repeat reached quant");
    typed_require(linear.weight()->data() == published->data() && linear.weight()->shape() == published->shape(), "repeat changed publication");
    typed_require(typed_read(runtime, linear.weight()) == expected_weight, "repeat changed markers");
}

void typed_throw_case(infinilm::runtime::Runtime &runtime) {
    auto quant = std::make_shared<CountedNone>(true);
    infinilm::nn::BaseLinear linear(runtime, 2, 3, quant, true, infinilm::runtime::DataType::kFloat32);
    typed_load(runtime, linear);
    infinilm::weights::ModelWeights *module = &linear;
    bool original_error = false;
    try {
        module->process_weights_after_loading(runtime, false);
    } catch (const FirstQuantError &error) { original_error = std::string(error.what()) == "typed-postload-first-quant-error"; }
    typed_require(original_error && quant->typed_calls == 1, "first quant exception replaced");
    typed_rejected([&] { module->process_weights_after_loading(runtime, true); });
    typed_require(quant->typed_calls == 1, "failed attempt latch did not stop re-entry");
    typed_require(typed_read(runtime, linear.weight()) == std::vector<float>{1, 2, 3, 4, 5, 6}, "throw-before-write changed markers");
}

void typed_preflight_case(infinilm::runtime::Runtime &runtime) {
    auto quant = std::make_shared<CountedNone>();
    infinilm::nn::BaseLinear linear(runtime, 2, 3, quant, true, infinilm::runtime::DataType::kFloat32);
    typed_load(runtime, linear);
    infinilm::weights::ModelWeights *module = &linear;
    auto wrong_device = infinilm::runtime::Runtime::create(Device{Device::Type::kCpu, 0});
    typed_rejected([&] { module->process_weights_after_loading(*wrong_device, false); });
    typed_require(quant->typed_calls == 0, "wrong preflight reached quant");
    module->process_weights_after_loading(runtime, false);
    typed_require(quant->typed_calls == 1, "wrong preflight consumed attempt");
}

void typed_no_matching_tls() {
    // Fresh thread proves no matching Context Runtime without relying on a setter.
    std::exception_ptr failure;
    std::thread worker([&] {
        try {
            const auto previous_type = infini::rt::runtime_device_type();
            int previous_index = 0;
            const bool had_device = infini::rt::runtime::GetDevice(&previous_index) == infini::rt::runtime::kSuccess;
            auto runtime = infinilm::runtime::Runtime::create(Device{Device::Type::kNvidia, 0});
            for (bool first : {false, true}) {
                for (bool repeat : {false, true}) {
                    typed_success_case(*runtime, first, repeat);
                }
            }
            typed_throw_case(*runtime);
            typed_preflight_case(*runtime);
            runtime->syncStream();
            typed_require(infini::rt::runtime_device_type() == previous_type, "typed path changed RT selection");
            if (had_device) {
                int current_index = -1;
                typed_require(infini::rt::runtime::GetDevice(&current_index) == infini::rt::runtime::kSuccess
                                  && current_index == previous_index,
                              "typed path changed RT device index");
            }
        } catch (...) { failure = std::current_exception(); }
    });
    worker.join();
    if (failure) {
        std::rethrow_exception(failure);
    }
    std::puts("typed post-load Module dispatch NONE false/true repeat/throw/preflight no-Runtime-binding PASS");
}

} // namespace

void check_inference_memory_config() {
    using namespace infinilm;
    using Backend = backends::AttentionBackend;
    config::ModelConfig model(nlohmann::json{{"torch_dtype", "bfloat16"}});
    auto chunked = [&](Device::Type device, int tp, int pp, Backend backend = Backend::PAGED_ATTN, bool capable = true) {
        return config::supports_chunked_prefill(model, device, tp, pp, backend, capable);
    };
    auto sharded = [&](Device::Type device, int tp, int pp) {
        return config::supports_tp_weight_sharding(model, device, tp, pp);
    };
    assert(chunked(Device::Type::kNvidia, 1, 1));
    assert(chunked(Device::Type::kNvidia, 2, 1));
    assert(chunked(Device::Type::kNvidia, 4, 1));
    assert(sharded(Device::Type::kNvidia, 2, 1));
    assert(sharded(Device::Type::kNvidia, 4, 1));
    assert(!sharded(Device::Type::kNvidia, 1, 1));
    assert(!chunked(Device::Type::kNvidia, 0, 1));
    assert(!chunked(Device::Type::kCpu, 2, 1));
    assert(!sharded(Device::Type::kCpu, 2, 1));
    assert(!chunked(Device::Type::kNvidia, 2, 2));
    assert(!sharded(Device::Type::kNvidia, 2, 2));
    assert(!chunked(Device::Type::kNvidia, 2, 1, Backend::STATIC_ATTN));
    assert(!chunked(Device::Type::kNvidia, 2, 1, Backend::PAGED_ATTN, false));
    for (const auto *dtype : {"float16", "float32"}) {
        model.get_config_json()["torch_dtype"] = dtype;
        assert(!chunked(Device::Type::kNvidia, 2, 1));
        assert(!sharded(Device::Type::kNvidia, 2, 1));
    }
    model.get_config_json()["torch_dtype"] = "bfloat16";
    for (const auto *name : {"llama", "qwen3", "unregistered_capability_probe"}) {
        model.get_config_json()["model_type"] = name;
        assert(chunked(Device::Type::kNvidia, 2, 1));
        assert(sharded(Device::Type::kNvidia, 2, 1));
    }
    auto quantized = config::ModelConfig(model.get_config_json());
    quantized.set_kv_quant_scheme(runtime::DataType::kInt8);
    assert(!config::supports_chunked_prefill(quantized, Device::Type::kNvidia, 2, 1, Backend::PAGED_ATTN, true));
    assert(config::supports_tp_weight_sharding(quantized, Device::Type::kNvidia, 2, 1));
    model.get_config_json()["quantization_config"] = {
        {"quant_method", "awq"}, {"bits", 4}, {"group_size", 128}, {"version", "GEMM"}, {"zero_point", true}};
    assert(!chunked(Device::Type::kNvidia, 2, 1));
    assert(!sharded(Device::Type::kNvidia, 2, 1));
    model.get_config_json().erase("quantization_config");
    assert(config::prefill_chunk_size(model) == 4096);
    assert(config::tp_vocab_sharding(model));
    assert(config::tp_embedding_sharding(model));
    for (const auto size : {0, 1, 128, 4096}) {
        model.get_config_json()["infinilm_prefill_chunk_size"] = size;
        assert(config::prefill_chunk_size(model) == static_cast<size_t>(size));
    }
    for (const auto &bad : std::vector<nlohmann::json>{-1, 1.5, "4096", nullptr, true, uint64_t{1} << 40}) {
        model.get_config_json()["infinilm_prefill_chunk_size"] = bad;
        bool rejected = false;
        try {
            (void)config::prefill_chunk_size(model);
        } catch (const std::invalid_argument &) { rejected = true; }
        assert(rejected);
    }
    model.get_config_json()["infinilm_tp_vocab_sharding"] = false;
    assert(!config::tp_vocab_sharding(model));
    assert(!config::tp_embedding_sharding(model));
    model.get_config_json()["infinilm_tp_embedding_sharding"] = true;
    assert(config::tp_embedding_sharding(model));
    model.get_config_json()["infinilm_tp_vocab_sharding"] = true;
    model.get_config_json()["infinilm_tp_embedding_sharding"] = false;
    assert(!config::tp_embedding_sharding(model));
    assert(config::tp_vocab_sharding(model));
    for (const auto *key : {"infinilm_tp_vocab_sharding", "infinilm_tp_embedding_sharding"}) {
        model.get_config_json()[key] = "true";
        bool rejected = false;
        try {
            if (std::string(key) == "infinilm_tp_vocab_sharding") {
                (void)config::tp_vocab_sharding(model);
            } else {
                (void)config::tp_embedding_sharding(model);
            }
        } catch (const std::invalid_argument &) { rejected = true; }
        assert(rejected);
        model.get_config_json()[key] = true;
    }
    using namespace layers::causal_lm_templates;
    struct UnknownModule {};
    using DenseDecoder = TextDecoderLayer<layers::attention::Attention, layers::mlp::MLP>;
    static_assert(ChunkedPrefillCapability<DenseDecoder>::value);
    static_assert(!ChunkedPrefillCapability<TextDecoderLayer<layers::attention::Attention, UnknownModule>>::value);
    static_assert(!ChunkedPrefillCapability<TextDecoderLayer<UnknownModule, layers::mlp::MLP>>::value);
    static_assert(ChunkedPrefillCapability<TextModel<DenseDecoder, true>>::value);
    static_assert(!ChunkedPrefillCapability<TextModel<DenseDecoder, false>>::value);
    static_assert(!ChunkedPrefillCapability<TextModel<UnknownModule, true>>::value);
    std::puts("Inference memory component capabilities and configuration PASS");
}

void check_constructed_memory_path(infinilm::runtime::Runtime &runtime,
                                   const infinilm::engine::distributed::RankInfo &rank) {
    using namespace infinilm;
    auto model_config = std::make_shared<config::ModelConfig>(nlohmann::json{
        {"model_type", "capability_probe"}, {"torch_dtype", "bfloat16"}, {"hidden_size", 256}, {"vocab_size", 128}, {"head_dim", 64}, {"intermediate_size", 512}, {"num_attention_heads", 4}, {"num_key_value_heads", 2}, {"num_hidden_layers", 1}, {"max_position_embeddings", 2048}, {"rms_norm_eps", 1e-6}, {"rope_theta", 10000.0}, {"attention_bias", false}, {"mlp_bias", false}});
    for (auto backend : {backends::AttentionBackend::PAGED_ATTN, backends::AttentionBackend::STATIC_ATTN}) {
        models::llama::LlamaForCausalLM model(runtime, model_config, rank, backend, layers::moe::EPConfig{});
        assert(model.prefill_chunk_size() == (backend == backends::AttentionBackend::PAGED_ATTN ? 4096 : 0));
    }
    std::puts("Constructed shared decoder reports actual memory execution path PASS");
}

int main(int argc, char **argv) {
    try {
        check_inference_memory_config();
        if (argc == 2 && std::string(argv[1]) == "--memory-config-only") {
            return 0;
        }
        check_forward_context_dispatch();
        if (argc == 2 && std::string(argv[1]) == "--forward-boundary-only") {
            return 0;
        }
        typed_no_matching_tls();
        auto runtime = infinilm::runtime::Runtime::create({Device::Type::kNvidia, 0});
        check_constructed_memory_path(*runtime, infinilm::engine::distributed::RankInfo(runtime->device()));
        auto settings = std::make_shared<infinilm::global_state::InfinilmConfig>();
        settings->attention_backend = infinilm::backends::AttentionBackend::STATIC_ATTN;
        infinilm::engine::distributed::RankInfo rank{runtime->device()};
        infinilm::global_state::ForwardContext forward_context;
        forward_context.runtime = runtime.get();
        typed_require(&forward_context.require_runtime() == runtime.get(),
                      "explicit ForwardContext Runtime changed");
        if (argc == 1) {
            check(false, *runtime, settings, rank);
            check(true, *runtime, settings, rank);
        } else {
            check(std::string(argv[1]) == "transpose", *runtime, settings, rank);
        }
        std::puts("post-load traversal PASS; unregistered fused weights preserved");
    } catch (const std::exception &error) {
        std::fprintf(stderr, "post-load traversal FAIL: %s\n", error.what());
        return 1;
    }
}
