#pragma once
#include "../../execution/submit.hpp"

#include "../../global_state/forward_context.hpp"
#include "runtime/runtime.hpp"

#include <cstddef>
#include <exception>
#include <stdexcept>
#include <utility>
#include <vector>

namespace infinilm::engine {

namespace detail {

inline void require_no_recurrent_prepare_state(const global_state::ForwardContext &context) {
    for (const auto &state : context.conv_state_vec) {
        if (state) {
            throw std::invalid_argument("prepare cannot isolate convolution state");
        }
    }
    for (const auto &state : context.ssm_state_vec) {
        if (state) {
            throw std::invalid_argument("prepare cannot isolate SSM state");
        }
    }
}

} // namespace detail

// Return only the static warmup write locations, retaining their original storage.
inline std::vector<infinilm::runtime::Tensor> static_prepare_views(
    infinilm::runtime::Runtime &runtime, const global_state::ForwardContext &context) {
    runtime.assertOwnerThread();
    detail::require_no_recurrent_prepare_state(context);
    if (context.kv_cache_vec.empty()) {
        throw std::invalid_argument("static prepare requires ordinary nonempty KV caches");
    }
    for (const auto &cache : context.kv_cache_vec) {
        if (!cache || cache->device() != runtime.device() || !cache->is_contiguous()
            || cache->ndim() != 5 || cache->size(0) != 2 || cache->size(1) == 0
            || cache->size(2) == 0 || cache->size(3) == 0 || cache->size(4) == 0) {
            throw std::invalid_argument("static prepare requires ordinary nonempty KV caches");
        }
    }
    std::vector<infinilm::runtime::Tensor> views;
    views.reserve(context.kv_cache_vec.size());
    for (const auto &cache : context.kv_cache_vec) {
        views.push_back(cache->narrow({{3, 0, 1}}));
    }
    return views;
}

// A paged warmup writes token_count slots in each of the first prepare_blocks
// physical blocks. Empty layer entries (e.g. pipeline-parallel layers) are skipped.
inline std::vector<infinilm::runtime::Tensor> paged_prepare_views(
    infinilm::runtime::Runtime &runtime, const global_state::ForwardContext &context,
    size_t num_blocks, size_t block_size, size_t prepare_blocks, size_t token_count) {
    runtime.assertOwnerThread();
    detail::require_no_recurrent_prepare_state(context);
    if (num_blocks == 0 || block_size == 0 || prepare_blocks == 0 || token_count == 0
        || prepare_blocks > num_blocks || token_count > block_size) {
        throw std::invalid_argument("paged prepare write region is out of bounds");
    }
    size_t active_layers = 0;
    for (const auto &cache : context.kv_cache_vec) {
        if (!cache) {
            continue;
        }
        if (cache->device() != runtime.device() || !cache->is_contiguous()
            || cache->ndim() != 5 || cache->size(0) != 2
            || cache->size(1) != num_blocks || cache->size(2) != block_size
            || cache->size(3) == 0 || cache->size(4) == 0) {
            throw std::invalid_argument("paged prepare requires ordinary contiguous KV caches");
        }
        ++active_layers;
    }
    if (active_layers == 0) {
        throw std::invalid_argument("paged prepare requires at least one KV cache");
    }
    std::vector<infinilm::runtime::Tensor> views;
    views.reserve(active_layers);
    for (const auto &cache : context.kv_cache_vec) {
        if (cache) {
            views.push_back(cache->narrow({{1, 0, prepare_blocks}, {2, 0, token_count}}));
        }
    }
    return views;
}

// Backup only the validated write views; the production cache allocation and
// addresses remain unchanged for graph capture. Call restore before publishing.
class PrepareStateGuard {
public:
    PrepareStateGuard(infinilm::runtime::Runtime &runtime,
                      global_state::ForwardContext &context,
                      std::vector<infinilm::runtime::Tensor> write_views)
        : runtime_(runtime), context_(context),
          attn_metadata_(context.attn_metadata),
          mamba_metadata_(context.mamba_metadata),
          mm_metadata_(context.mm_metadata) {
        runtime_.assertOwnerThread();
        if (runtime_.isGraphRecording()) {
            throw std::logic_error("prepare backup must run outside graph recording");
        }
        if (write_views.empty()) {
            throw std::invalid_argument("prepare requires at least one write view");
        }
        for (const auto &view : write_views) {
            if (!view || view->device() != runtime_.device() || view->numel() == 0) {
                throw std::invalid_argument("prepare requires nonempty local write views");
            }
        }
        saved_views_.reserve(write_views.size());
        try {
            for (auto &view : write_views) {
                auto saved = infinilm::runtime::Tensor::empty(runtime_, view->shape(), view->dtype());
                // Keep both buffers alive even if rearrange_ enqueues work then throws.
                saved_views_.push_back({std::move(view), std::move(saved)});
                const auto &entry = saved_views_.back();
                infinilm::execution::submit<infini::ops::Copy>(
                    runtime_, infini::ops::Copy::DefaultConfig(runtime_.device().type()),
                    entry.view, false, entry.saved);
            }
            runtime_.syncStream();
        } catch (...) {
            // Pending device copies may still reference the saved buffers.
            try {
                runtime_.syncStream();
            } catch (...) {
                std::terminate();
            }
            throw;
        }
        active_ = true;
    }

    PrepareStateGuard(const PrepareStateGuard &) = delete;
    PrepareStateGuard &operator=(const PrepareStateGuard &) = delete;

    ~PrepareStateGuard() noexcept {
        if (active_) {
            try {
                restore();
            } catch (...) {
                std::terminate();
            }
        }
    }

    void restore() {
        if (!active_) {
            return;
        }
        // Do not retry a partially failed restore while unwinding.
        active_ = false;
        try {
            for (const auto &entry : saved_views_) {
                infinilm::execution::submit<infini::ops::Copy>(
                    runtime_, infini::ops::Copy::DefaultConfig(runtime_.device().type()),
                    entry.saved, false, entry.view);
            }
            runtime_.syncStream();
        } catch (...) {
            // A failed copy may still be in flight; drain before freeing backup.
            try {
                runtime_.syncStream();
            } catch (...) {
                std::terminate();
            }
            // Device restore failure remains fatal to this prepare. Restore
            // metadata as well, then propagate the original copy error.
            context_.attn_metadata = std::move(attn_metadata_);
            context_.mamba_metadata = std::move(mamba_metadata_);
            context_.mm_metadata = std::move(mm_metadata_);
            throw;
        }
        context_.attn_metadata = std::move(attn_metadata_);
        context_.mamba_metadata = std::move(mamba_metadata_);
        context_.mm_metadata = std::move(mm_metadata_);
    }

private:
    infinilm::runtime::Runtime &runtime_;
    global_state::ForwardContext &context_;
    global_state::AttentionMetadata attn_metadata_;
    global_state::MambaMetadata mamba_metadata_;
    global_state::MultiModalMetadata mm_metadata_;
    struct SavedView {
        infinilm::runtime::Tensor view;
        infinilm::runtime::Tensor saved;
    };
    std::vector<SavedView> saved_views_;
    bool active_ = false;
};

} // namespace infinilm::engine
