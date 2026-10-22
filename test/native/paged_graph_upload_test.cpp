#include "../../csrc/runtime/graph_metadata_upload_lease.hpp"
#include <cassert>
#include <functional>
#include <iostream>
struct Memory {
    std::vector<std::byte> storage;
    explicit Memory(size_t n) : storage(n) {}
    std::byte *data() { return storage.data(); }
    size_t size() { return storage.size(); }
};
struct Runtime {
    std::vector<std::function<void()>> queue;
    size_t syncs = 0, allocs = 0, h2d = 0, d2d = 0;
    bool fail = false;
    void assertOwnerThread() {}
    void syncStream() {
        ++syncs;
        for (auto &f : queue) {
            f();
        }
        queue.clear();
    }
    auto allocatePinnedHostMemory(size_t n) {
        ++allocs;
        return std::make_shared<Memory>(n);
    }
    auto allocateExactMemory(size_t n) {
        ++allocs;
        return std::make_shared<Memory>(n);
    }
    void memcpyH2D(void *d, const void *s, size_t n, bool async) {
        assert(async);
        ++h2d;
        queue.push_back([=] { std::memcpy(d, s, n); });
        if (fail) {
            throw std::runtime_error("submission failed after enqueue");
        }
    }
    void memcpyD2D(void *d, const void *s, size_t n, bool async) {
        assert(async);
        ++d2d;
        queue.push_back([=] { std::memcpy(d, s, n); });
    }
};
using Lease = infinilm::runtime::GraphMetadataUploadLease<Runtime, Memory>;
int main() {
    Runtime runtime;
    int dst1 = 0, dst2 = 0;
    {
        Lease upload(runtime);
        {
            int a = 7, b = 9;
            upload.upload({{&dst1, &a, sizeof(a)}, {&dst2, &b, sizeof(b)}});
            a = b = -1;
        }
        assert(runtime.syncs == 0 && runtime.h2d == 1 && runtime.d2d == 2);
        int next = 11;
        upload.upload({{&dst1, &next, sizeof(next)}});
        assert(dst1 == 7 && dst2 == 9 && runtime.syncs == 1 && runtime.allocs == 2);
        upload.drain();
        assert(dst1 == 11 && upload.reuse_syncs == 2);
        upload.drain();
        assert(runtime.syncs == 2);
        std::vector<int> big(100, 42), out(100);
        upload.upload({{out.data(), big.data(), big.size() * sizeof(int)}});
        upload.drain();
        assert(out == big && runtime.allocs == 4);
    }
    assert(runtime.queue.empty());
    Runtime failing;
    {
        Lease upload(failing);
        failing.fail = true;
        int value = 5;
        try {
            upload.upload({{&dst1, &value, sizeof(value)}});
            assert(false);
        } catch (const std::runtime_error &) {}
    }
    assert(failing.queue.empty() && failing.syncs == 1);
    Runtime destructor;
    {
        Lease upload(destructor);
        int value = 23;
        upload.upload({{&dst1, &value, sizeof(value)}});
    }
    assert(dst1 == 23 && destructor.queue.empty());
    std::cout << "paged graph staging lifetime, reuse, growth, drain and enqueue failure passed\n";
}
