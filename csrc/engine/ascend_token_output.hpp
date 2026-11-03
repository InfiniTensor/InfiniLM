#pragma once

#include <cstring>
#include <infinicore/context/context.hpp>
#include <infinicore/tensor.hpp>
#include <infinirt.h>
#include <stdexcept>

namespace infinilm::engine {

// Sampling is complete before this returns. Each returned CPU tensor owns its
// data; the per-thread pinned transfer buffer is never exposed to the caller.
inline infinicore::Tensor copy_sampled_tokens_to_host(
    const infinicore::Tensor &tokens, bool enable_async) {
    using namespace infinicore;
    if (!tokens) {
        throw std::invalid_argument("sampled token tensor must not be null");
    }
    const bool eligible = enable_async
                       && tokens->device().getType() == Device::Type::ASCEND
                       && tokens->dtype() == DataType::I64
                       && tokens->is_contiguous()
                       && tokens->numel() > 0 && tokens->numel() <= 16;
    if (!eligible) {
        context::syncStream();
        auto host = tokens->to(Device::cpu());
        context::syncStream();
        return host;
    }
    struct Staging {
        Device owner = Device::cpu();
        Tensor buffer;
    };
    static thread_local Staging staging;
    context::setDevice(tokens->device());
    if (!staging.buffer || staging.owner != tokens->device()) {
        // The previous transfer is synchronized below before every return.
        staging.buffer = Tensor::empty({16}, DataType::I64, Device::cpu(), true);
        staging.owner = tokens->device();
    }
    const auto status = infinirtMemcpyAsync(
        staging.buffer->data(), tokens->data(), tokens->nbytes(),
        INFINIRT_MEMCPY_D2H, context::getStream());
    if (status != INFINI_STATUS_SUCCESS) {
        throw std::runtime_error("stream-ordered sampled-token D2H failed");
    }
    // Unlike synchronous aclrtMemcpy, this transfer follows the queued sampler
    // on the same stream. One synchronization makes both producer and copy
    // visible to the host without reading the previous step's token.
    context::syncStream();
    auto host = Tensor::empty(tokens->shape(), tokens->dtype(), Device::cpu());
    std::memcpy(host->data(), staging.buffer->data(), tokens->nbytes());
    context::setDevice(tokens->device());
    return host;
}

} // namespace infinilm::engine
