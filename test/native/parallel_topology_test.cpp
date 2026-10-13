#ifdef NDEBUG
#undef NDEBUG
#endif

#include "../../csrc/parallel/topology.hpp"
#include "../../csrc/engine/distributed/dist_config.hpp"

#include <cassert>
#include <iostream>
#include <limits>
#include <stdexcept>

using infinilm::parallel::Topology;
using infinilm::engine::distributed::DistConfig;

template <typename Exception, typename Function>
void expect_throw(Function action) {
    bool threw = false;
    try { action(); } catch (const Exception &) { threw = true; }
    assert(threw);
}

int main() {
    expect_throw<std::invalid_argument>([] { Topology(0); });
    expect_throw<std::invalid_argument>([] { Topology(1, 0); });
    expect_throw<std::invalid_argument>([] { Topology(1, 2, -1); });
    expect_throw<std::invalid_argument>([] { Topology(1, 2, 2); });
    const size_t rank_limit = static_cast<size_t>(std::numeric_limits<int>::max());
    expect_throw<std::invalid_argument>([=] { Topology(rank_limit + 1); });
    expect_throw<std::invalid_argument>([=] { Topology(rank_limit, 2); });
    expect_throw<std::invalid_argument>([] { DistConfig(-1); });
    expect_throw<std::invalid_argument>([] { DistConfig(0); });
    assert(DistConfig().tp_device_ids == std::vector<int>{0});
    assert(DistConfig(3).tp_device_ids == (std::vector<int>{0, 1, 2}));
    assert(DistConfig(std::vector<int>{3, 1}).tp_device_ids == (std::vector<int>{3, 1}));

    size_t mapping_cases = 0;
    size_t range_cases = 0;
    for (int tp : {1, 2, 4, 8}) {
        for (int pp = 1; pp <= 8; ++pp) {
            for (int stage = 0; stage < pp; ++stage) {
                const Topology topology(static_cast<size_t>(tp), pp, stage);
                assert(topology.tp_size() == tp && topology.pp_size() == pp);
                assert(topology.pp_stage() == stage && topology.world_size() == tp * pp);
                for (int rank = 0; rank < tp; ++rank) {
                    assert(topology.world_rank(rank) == stage * tp + rank);
                    ++mapping_cases;
                }
                expect_throw<std::out_of_range>([&] { topology.world_rank(-1); });
                expect_throw<std::out_of_range>([&] { topology.world_rank(tp); });
                for (size_t layers = 0; layers <= 128; ++layers) {
                    const auto range = topology.layer_range(layers);
                    assert(range.first == layers * static_cast<size_t>(stage) / static_cast<size_t>(pp));
                    assert(range.second == layers * static_cast<size_t>(stage + 1) / static_cast<size_t>(pp));
                    ++range_cases;
                }
            }
        }
    }
    // Cover overflow-sized total_layers without requiring a huge model allocation.
    for (int pp = 1; pp <= 8; ++pp) {
        const size_t layers = std::numeric_limits<size_t>::max();
        size_t next = 0;
        for (int stage = 0; stage < pp; ++stage) {
            const auto range = Topology(1, pp, stage).layer_range(layers);
            assert(range.first == next && range.second >= range.first);
            assert(range.second <= layers);
            next = range.second;
        }
        assert(next == layers);
    }
    assert(Topology(rank_limit).world_size() == std::numeric_limits<int>::max());
    std::cout << "parallel topology host PASS mappings=" << mapping_cases
              << " ranges=" << range_cases << '\n';
}
