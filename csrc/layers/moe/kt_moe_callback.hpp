#pragma once
#include "infinicore/tensor.hpp"
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace infinilm::layers::moe {

// Global registry of KTransformers MoE callbacks (one per layer_idx).
//
// Concurrency contract:
//  - Callbacks are stored as immutable shared_ptr entries. get() copies the
//    entry out under the lock and the user callback is invoked WITHOUT any
//    lock held, so a callback that acquires the GIL can never deadlock
//    against set()/clear() called from a GIL-holding thread.
//  - get() is a single atomic lookup: no has()+call() TOCTOU window.
class KTMoECallbackRegistry {
public:
    using CallbackFn = std::function<infinicore::Tensor(
        const infinicore::Tensor &hidden_states,
        const infinicore::Tensor &topk_weights,
        const infinicore::Tensor &topk_ids,
        int layer_idx)>;

    static KTMoECallbackRegistry &instance() {
        static KTMoECallbackRegistry reg;
        return reg;
    }

    void set(int layer_idx, CallbackFn cb) {
        std::lock_guard<std::mutex> lk(mtx_);
        cbs_[layer_idx] = std::make_shared<const CallbackFn>(std::move(cb));
    }

    void clear() {
        std::lock_guard<std::mutex> lk(mtx_);
        cbs_.clear();
    }

    // Returns nullptr when no callback is registered for this layer.
    // The returned pointer stays valid even if set/clear run concurrently.
    std::shared_ptr<const CallbackFn> get(int layer_idx) {
        std::lock_guard<std::mutex> lk(mtx_);
        auto it = cbs_.find(layer_idx);
        return it == cbs_.end() ? nullptr : it->second;
    }

    bool empty() {
        std::lock_guard<std::mutex> lk(mtx_);
        return cbs_.empty();
    }

private:
    std::mutex mtx_;
    std::unordered_map<int, std::shared_ptr<const CallbackFn>> cbs_;
};

} // namespace infinilm::layers::moe
