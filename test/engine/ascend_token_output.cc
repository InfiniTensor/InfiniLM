#include "csrc/engine/ascend_token_output.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <infinicore/ops/argmax.hpp>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

extern "C" int check_async_token_output() {
    using namespace infinicore;
    std::array<std::string, 4> errors;
    std::mutex printing;
    std::vector<std::thread> workers;
    for (int rank = 0; rank < 4; ++rank) {
        workers.emplace_back([&, rank] {
            try {
                Device dev(Device::Type::ASCEND, rank);
                context::setDevice(dev);
                constexpr size_t vocab = 73448;
                for (size_t batch : {1, 4, 8, 16, 17}) {
                    auto scores = Tensor::empty({batch, vocab}, DataType::BF16, dev);
                    std::vector<uint16_t> values(batch * vocab);
                    Tensor first;
                    std::vector<int64_t> first_values;
                    for (int step = 0; step < 32; ++step) {
                        std::fill(values.begin(), values.end(), uint16_t(0));
                        std::vector<int64_t> expected(batch);
                        for (size_t row = 0; row < batch; ++row) {
                            size_t index = (step * 7919 + row * 31 + rank) % (vocab - 1);
                            values[row * vocab + index] = 0x4120;     // BF16 10
                            values[row * vocab + index + 1] = 0x4120; // first-index tie
                            expected[row] = index;
                        }
                        context::setDevice(dev);
                        context::memcpyH2D(scores->data(), values.data(), values.size() * 2, false);
                        auto selected = op::argmax(scores, 1); // asynchronous producer
                        auto result = infinilm::engine::copy_sampled_tokens_to_host(selected, true);
                        if (context::getDevice() != dev) {
                            throw std::runtime_error("sampled-token copy changed the caller device context");
                        }
                        const auto *actual = reinterpret_cast<const int64_t *>(result->data());
                        if (!std::equal(expected.begin(), expected.end(), actual)) {
                            throw std::runtime_error("stale or incorrect sampled tokens");
                        }
                        if (step == 0) {
                            first = result;
                            first_values = expected;
                        }
                    }
                    const auto *retained = reinterpret_cast<const int64_t *>(first->data());
                    if (!std::equal(first_values.begin(), first_values.end(), retained)) {
                        throw std::runtime_error("returned CPU snapshot overwritten by staging buffer");
                    }
                    auto measure = [&](bool enabled) {
                        context::syncStream();
                        auto start = std::chrono::steady_clock::now();
                        for (int i = 0; i < 100; ++i) {
                            auto selected = op::argmax(scores, 1);
                            auto result = infinilm::engine::copy_sampled_tokens_to_host(selected, enabled);
                        }
                        return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / 100;
                    };
                    std::vector<double> old_times, new_times;
                    measure(false);
                    measure(true);
                    for (int rep = 0; rep < 8; ++rep) {
                        if (rep % 2) {
                            new_times.push_back(measure(true));
                            old_times.push_back(measure(false));
                        } else {
                            old_times.push_back(measure(false));
                            new_times.push_back(measure(true));
                        }
                    }
                    std::sort(old_times.begin(), old_times.end());
                    std::sort(new_times.begin(), new_times.end());
                    std::lock_guard<std::mutex> lock(printing);
                    std::cout << "ASYNC_TOKEN_PASS rank=" << rank << " batch=" << batch
                              << " changing_inputs=32 retained_snapshot=1 ties=first"
                              << " baseline_us=" << (old_times[3] + old_times[4]) / 2
                              << " async_us=" << (new_times[3] + new_times[4]) / 2 << std::endl;
                }
            } catch (std::exception const &e) {
                errors[rank] = e.what();
            }
        });
    }
    for (auto &worker : workers) {
        worker.join();
    }
    int failures = 0;
    for (int rank = 0; rank < 4; ++rank) {
        if (!errors[rank].empty()) {
            std::cerr << "ASYNC_TOKEN_FAIL " << rank << " " << errors[rank] << std::endl;
            ++failures;
        }
    }
    return failures;
}
