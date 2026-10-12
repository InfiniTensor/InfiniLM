#pragma once
#include "../linear/base_linear.hpp"
#include "runtime/runtime.hpp"
#include <exception>
#include <stdexcept>
#include <utility>
#include <vector>

namespace infinilm::quantization {
// Ephemeral homogeneous owner transaction; not Ready, a loader or a cache.
class W4ModelPlan final {
public:
    explicit W4ModelPlan(QuantScheme expected_scheme,
                         std::vector<std::shared_ptr<nn::BaseLinear>> layers,
                         std::shared_ptr<nn::BaseLinear> dense_head = {})
        : expected_scheme_(expected_scheme), dense_head_(std::move(dense_head)) {
        if (expected_scheme_ != QuantScheme::AWQ_W4A16 && expected_scheme_ != QuantScheme::GPTQ_W4A16)
            throw std::invalid_argument("W4 plan requires AWQ or GPTQ scheme");
        if (dense_head_ && dense_head_->get_quantization()->get_quant_scheme() != QuantScheme::NONE)
            throw std::invalid_argument("W4 dense head exemption is NONE only");
        entries_.reserve(layers.size());
        for (auto& layer : layers) {
            if (!layer || layer->get_quantization()->get_quant_scheme() != expected_scheme_)
                throw std::invalid_argument("W4 plan rejects null/mixed layer");
            for (const auto& entry : entries_)
                if (entry.layer.get() == layer.get()) throw std::invalid_argument("W4 duplicate layer owner");
            entries_.push_back({std::move(layer), {}});
        }
    }
    void validate_metadata(infinilm::runtime::Runtime& runtime) const {
        runtime.assertOwnerThread();
        for (const auto& entry : entries_) {
            if (expected_scheme_ == QuantScheme::AWQ_W4A16) {
                entry.layer->validate_awq_metadata(runtime);
            } else {
                const auto quant = entry.layer->get_quantization();
                const auto group = quant->get_or<int>("group_size", quant->get_or<int>("q_group_size", 128));
                if (entry.layer->dtype() != infinilm::runtime::DataType::kFloat16 ||
                    quant->get_or<int>("bits", quant->get_or<int>("w_bit", 4)) != 4 ||
                    (group != 32 && group != 128) ||
                    !quant->get_or<bool>("sym", true) ||
                    quant->get_or<std::string>("checkpoint_format", "gptq") != "gptq" ||
                    quant->get_or<bool>("desc_act", false))
                    throw std::invalid_argument("GPTQ owner requires FP16 canonical symmetric historical g32/g128 non-desc");
                (void)entry.layer->validate_gptq_metadata(runtime);
            }
        }
    }
    void prepare(infinilm::runtime::Runtime& runtime) {
        runtime.assertOwnerThread();
        if (prepared_ || sealed_) throw std::logic_error("W4 model preparation is single-use");
        for (auto& entry : entries_) {
            if (entry.candidate) throw std::logic_error("W4 partial preparation cannot be retried");
            entry.candidate = entry.layer->prepare_w4_after_loading(runtime, false);
        }
        // Only exact root head, validated pre_transpose=false; no hidden noarg hook.
        if (dense_head_) dense_head_->process_weights_after_loading(runtime, false);
        prepared_ = true;
    }
    // Worker calls only after successful owner sync AND trim.
    void seal() {
        if (!prepared_ || sealed_) throw std::logic_error("W4 model plan is not fully prepared");
        for (const auto& entry : entries_)
            if (!entry.candidate) throw std::logic_error("W4 missing prepared candidate");
        sealed_ = true;
    }
    void commit() noexcept {
        if (!sealed_) std::terminate();
        for (auto& entry : entries_)
            entry.layer->publish_w4_after_loading(std::move(entry.candidate));
        sealed_ = false;
    }
private:
    struct Entry {
        std::shared_ptr<nn::BaseLinear> layer;
        std::shared_ptr<const W4Execution> candidate;
    };
    const QuantScheme expected_scheme_;
    std::vector<Entry> entries_;
    std::shared_ptr<nn::BaseLinear> dense_head_;
    bool prepared_ = false, sealed_ = false;
};
} // namespace infinilm::quantization
