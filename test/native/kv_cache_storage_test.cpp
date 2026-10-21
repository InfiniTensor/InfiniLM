#include "cache/kv_cache.hpp"
#include "engine/distributed/communication_group.hpp"
#include "runtime/runtime.hpp"
#include "runtime/tensor.hpp"

#include <cstdint>
#include <cstdio>
#include <stdexcept>

namespace {
using namespace infinilm;

void require(bool value, const char *message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

void check_storage(runtime::Runtime &owner, runtime::Tensor cache) {
    const auto bytes = cache->nbytes();
    // These cache shapes require 32.5 MiB, which used to reserve a 64 MiB block.
    require(bytes == 325 * 1024 * 1024 / 10, "unexpected cache geometry");
    owner.syncStream();
    uint16_t endpoints[2] = {1, 1};
    owner.memcpyD2H(&endpoints[0], cache->data(), sizeof(uint16_t));
    owner.memcpyD2H(&endpoints[1], cache->data() + bytes - sizeof(uint16_t), sizeof(uint16_t));
    require(endpoints[0] == 0 && endpoints[1] == 0, "cache must remain zero initialized");

    owner.startGraphRecording();
    auto blob = cache->to_blob_(owner);
    require(owner.graphPinnedMemoryBytes() == bytes, "KV backing was rounded to a size class");
    auto alias = blob->resume_from_blob_(owner);
    require(alias->data() == cache->data(), "cache graph alias lost its address");
    auto graph = owner.stopGraphRecording();
    cache.reset();
    blob.reset();
    alias.reset();
    owner.trimMemory();
    require(owner.graphPinnedMemoryBytes() == bytes, "trim released graph-pinned KV backing");
    graph->run();
    graph.reset();
    require(owner.graphPinnedMemoryBytes() == 0, "graph release retained KV pin");
    owner.trimMemory();
    owner.syncStream();
}
} // namespace

int main() {
    try {
        auto owner = runtime::Runtime::create({runtime::Device::Type::kNvidia, 0});
        engine::distributed::RankInfo rank(owner->device());
        rank.tp_size = 2;
        const cache::PagedKVCacheConfig paged(65, 256, 1);
        check_storage(*owner, cache::PagedKVCache::create_layer_kv_cache(
                                  *owner, rank, 128, 128, 8, 8, runtime::DataType::kBFloat16, paged));
        const cache::StaticKVCacheConfig fixed(1, 16640);
        check_storage(*owner, cache::StaticKVCache::create_layer_kv_cache(
                                  *owner, rank, 128, 128, 8, 8, 40960, runtime::DataType::kBFloat16, fixed));
    } catch (const std::exception &error) {
        std::fprintf(stderr, "kv_cache_storage_test: %s\n", error.what());
        return 1;
    }
    std::puts("kv_cache_storage_test: PASS");
    return 0;
}
