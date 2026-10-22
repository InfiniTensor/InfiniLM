#include "../../csrc/engine/compiler/paged_decode_bucket.hpp"
#include <cassert>
#include <iostream>
using namespace infinilm::engine;
int main() {
    std::vector<std::pair<size_t, size_t>> profiles;
    for (size_t b : {1, 8, 16}) {
        for (size_t k : {2048, 4096, 0}) { profiles.emplace_back(b, k); }
    }
    for (size_t b : {1, 8, 16}) {
        for (size_t actual : {1, 2048, 2049, 4096, 4097, 16384}) {
            auto chosen = select_graph_capture_profile(profiles, b, actual, 32768);
            assert(chosen && *chosen == (actual <= 2048 ? 2048 : actual <= 4096 ? 4096
                                                                                : 0));
        }
    }
    assert(!select_graph_capture_profile(profiles, 2, 1, 32768));
    assert(!select_graph_capture_profile(profiles, 1, 32769, 32768));
    assert(!select_graph_capture_profile({{8, 2048}}, 8, 2049, 32768));
    assert(!select_graph_capture_profile({}, 1, 1, 32768));
    bool invalid = false;
    try {
        validate_graph_capture_profile(17, 2048, 16, 256, 128);
    } catch (const std::invalid_argument &) { invalid = true; }
    assert(invalid);
    for (int b : {1, 8, 16}) {
        std::vector<int32_t> lengths(b, 2048), table(b * 128, -1);
        for (int row = 0; row < b; ++row) {
            for (int col = 0; col < 8; ++col) {
                table[row * 128 + col] = col;
            }
        }
        auto normalized = normalize_graph_page_table(lengths.data(), b, table.data(), 128, 8, 256, 128);
        assert(normalized && normalized->size() == size_t(b * 8));
        lengths[0] = 2049;
        assert(!normalize_graph_page_table(lengths.data(), b, table.data(), 128, 8, 256, 128));
        lengths[0] = 2048;
        table[0] = 128;
        assert(!normalize_graph_page_table(lengths.data(), b, table.data(), 128, 8, 256, 128));
    }
    std::cout << "paged graph profile boundaries, batch transitions and invalid tables passed\n";
}
