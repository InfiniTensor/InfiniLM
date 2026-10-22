#include "runtime/graph_metadata_upload.hpp"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace infinilm::runtime;
int main() {
    for (int rank = 0; rank < 2; ++rank) {
        auto runtime = Runtime::create(Device{Device::Type::kNvidia, rank});
        runtime->activate();
        auto first = runtime->allocateExactMemory(8192);
        auto second = runtime->allocateExactMemory(8192);
        GraphMetadataUpload upload(*runtime);
        for (int step = 0; step < 100; ++step) {
            const size_t n = step < 50 ? 16 : 1024;
            std::vector<int32_t> a(n, step), b(n, step + rank + 123);
            upload.upload({{first->data(), a.data(), n * sizeof(int32_t)},
                           {second->data(), b.data(), n * sizeof(int32_t)}});
            std::fill(a.begin(), a.end(), -1);
            std::fill(b.begin(), b.end(), -1);
            std::vector<int32_t> result(n);
            runtime->memcpyD2H(result.data(), first->data(), n * sizeof(int32_t));
            if (!std::all_of(result.begin(), result.end(), [step](auto v) { return v == step; })) {
                throw std::runtime_error("first staging destination mismatch");
            }
            runtime->memcpyD2H(result.data(), second->data(), n * sizeof(int32_t));
            if (!std::all_of(result.begin(), result.end(), [step, rank](auto v) { return v == step + rank + 123; })) {
                throw std::runtime_error("second staging destination mismatch");
            }
        }
        upload.drain();
        if (upload.uploads != 100) {
            throw std::runtime_error("wrong upload count");
        }
        std::cout << "rank=" << rank << " staging_uploads=" << upload.uploads
                  << " drains=" << upload.reuse_syncs << " passed\n";
    }
}
