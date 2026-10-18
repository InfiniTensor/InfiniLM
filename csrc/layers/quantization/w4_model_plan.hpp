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
                         std::vector<std::shared_ptr<nn::BaseLinear>> dense_head = {},
                         std::vector<std::shared_ptr<weights::ModelWeights>> hooks = {})
        : expected_scheme_(expected_scheme), dense_head_(std::move(dense_head)), hooks_(std::move(hooks)) {
        if (expected_scheme_ != QuantScheme::AWQ_W4A16 && expected_scheme_ != QuantScheme::GPTQ_W4A16) {
            throw std::invalid_argument("W4 plan requires AWQ or GPTQ scheme");
        }
        for (const auto &head : dense_head_) {
            if (!head || head->get_quantization()->get_quant_scheme() != QuantScheme::NONE) {
                throw std::invalid_argument("W4 dense exemption is NONE only");
            }
        }
        entries_.reserve(layers.size());
        for (auto &layer : layers) {
            if (!layer || layer->get_quantization()->get_quant_scheme() != expected_scheme_) {
                throw std::invalid_argument("W4 plan rejects null/mixed layer");
            }
            for (const auto &entry : entries_) {
                if (entry.layer.get() == layer.get()) {
                    throw std::invalid_argument("W4 duplicate layer owner");
                }
            }
            entries_.push_back({std::move(layer), {}});
        }
    }
    void validate_metadata(infinilm::runtime::Runtime &runtime) const {
        runtime.assertOwnerThread();
        for (const auto &entry : entries_) {
            if (expected_scheme_ == QuantScheme::AWQ_W4A16) {
                entry.layer->validate_awq_metadata(runtime);
            } else {
                (void)entry.layer->validate_gptq_metadata(runtime);
            }
        }
    }
    void prepare(infinilm::runtime::Runtime &runtime) {
        runtime.assertOwnerThread();
        if (prepared_ || sealed_) {
            throw std::logic_error("W4 model preparation is single-use");
        }
        for (auto &entry : entries_) {
            if (entry.candidate) {
                throw std::logic_error("W4 partial preparation cannot be retried");
            }
            entry.candidate = entry.layer->prepare_w4_after_loading(runtime, false);
        }
        // Dense owners use the same validated post-load transpose policy.
        for (const auto &head : dense_head_) {
            head->process_weights_after_loading(runtime, false);
        }
        for (const auto &owner : hooks_) {
            owner->process_weights_after_loading(runtime, false);
        }
        prepared_ = true;
    }
    // Worker calls only after successful owner sync AND trim.
    void seal() {
        if (!prepared_ || sealed_) {
            throw std::logic_error("W4 model plan is not fully prepared");
        }
        for (const auto &entry : entries_) {
            if (!entry.candidate) {
                throw std::logic_error("W4 missing prepared candidate");
            }
        }
        sealed_ = true;
    }
    void commit() noexcept {
        if (!sealed_) {
            std::terminate();
        }
        for (auto &entry : entries_) {
            entry.layer->publish_w4_after_loading(std::move(entry.candidate));
        }
        sealed_ = false;
    }

private:
    struct Entry {
        std::shared_ptr<nn::BaseLinear> layer;
        std::shared_ptr<const W4Execution> candidate;
    };
    const QuantScheme expected_scheme_;
    std::vector<Entry> entries_;
    std::vector<std::shared_ptr<nn::BaseLinear>> dense_head_;
    std::vector<std::shared_ptr<weights::ModelWeights>> hooks_;
    bool prepared_ = false, sealed_ = false;
};
} // namespace infinilm::quantization
