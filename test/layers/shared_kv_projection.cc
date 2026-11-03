#include "csrc/global_state/global_state.hpp"
#include "csrc/layers/linear/fused_linear.hpp"
#include <array>
#include <cstring>
#include <infinicore/context/context.hpp>
#include <iostream>
#include <thread>
#include <vector>
using namespace infinicore;
using namespace infinilm;
static uint16_t bf(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    u += 0x7fff + ((u >> 16) & 1);
    return u >> 16;
}
static float fp(uint16_t b) {
    uint32_t u = uint32_t(b) << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
extern "C" int check_shared_kv_projection(int bad_case) {
    std::array<std::string, 4> errors;
    std::vector<std::thread> workers;
    for (int rank = 0; rank < 4; ++rank) {
        workers.emplace_back([&, rank] {
            try {
                Device dev(Device::Type::ASCEND, rank);
                context::setDevice(dev);
                auto cfg = std::make_shared<global_state::InfinilmConfig>(backends::AttentionBackend::FLASH_ATTN, nullptr);
                cfg->pre_transpose = true;
                global_state::initialize_infinilm_config(cfg);
                engine::distributed::RankInfo ri(dev);
                ri.tp_size = 4;
                ri.tp_rank = rank;
                global_state::initialize_model_parallel(ri);
                const bool biased = bad_case != 3;
                layers::linear::QKVParallelLinear layer(256, 128, 64, 64, nullptr, biased, DataType::BF16, dev, ri);
                std::vector<uint16_t> w(6144 * 256), bias(6144);
                for (int row = 0; row < 6144; ++row) {
                    int part = row / 2048, local = row % 2048, head = local / 128, dim = local % 128, logical = part ? head / 8 : head;
                    bias[row] = bf(float((logical + dim % 3) % 9 - 4) / 64.f);
                    for (int col = 0; col < 256; ++col) {
                        w[row * 256 + col] = bf(float((logical * 3 + dim % 5 + col % 11 + rank) % 31 - 15) / 1024.f);
                    }
                }
                if (bad_case == 1) {
                    w[(2048 + 128) * 256] = bf(.5f);
                }
                if (bad_case == 2) {
                    bias[2048 + 128] = bf(.5f);
                }
                context::memcpyH2D(layer.weight()->data(), w.data(), w.size() * 2, false);
                if (biased) {
                    context::memcpyH2D(layer.bias()->data(), bias.data(), bias.size() * 2, false);
                }
                context::syncStream();
                layer.process_weights_after_loading();
                for (int T : {1, 16, 17, 41, 265, 1024, 2047, 2048}) {
                    auto input = Tensor::empty({1, size_t(T), 256}, DataType::BF16, dev);
                    std::vector<uint16_t> x(T * 256);
                    for (size_t j = 0; j < x.size(); ++j) {
                        x[j] = bf(float(int(j % 17) - 8) / 16.f);
                    }
                    context::memcpyH2D(input->data(), x.data(), x.size() * 2, false);
                    context::startGraphRecording();
                    auto [q, k, v, qk] = layer.forward_split_with_qk(input);
                    auto got = Tensor::empty({1, size_t(T), 6144}, DataType::BF16, dev);
                    got->narrow({{2, 0, 2048}})->copy_from(q);
                    got->narrow({{2, 2048, 2048}})->copy_from(k);
                    got->narrow({{2, 4096, 2048}})->copy_from(v);
                    auto graph = context::stopGraphRecording();
                    for (int phase = 0; phase < 5; ++phase) {
                        for (size_t j = 0; j < x.size(); ++j) {
                            x[j] = bf(float(int((j + phase) % 17) - 8) / 16.f);
                        }
                        context::memcpyH2D(input->data(), x.data(), x.size() * 2, false);
                        graph->run();
                        context::syncStream();
                        std::vector<uint16_t> a(got->numel());
                        context::memcpyD2H(a.data(), got->data(), a.size() * 2);
                        auto expected = layer.forward(input);
                        context::syncStream();
                        std::vector<uint16_t> ref(a.size());
                        context::memcpyD2H(ref.data(), expected->data(), ref.size() * 2);
                        if (a != ref) {
                            throw std::runtime_error("full QKV output differs from original: T=" + std::to_string(T));
                        }
                        for (int token : {0, T / 2, T - 1}) {
                            for (int col : {0, 127, 2048, 2176, 4096, 6143}) {
                                float sum = 0;
                                for (int j = 0; j < 256; ++j) {
                                    sum += fp(x[token * 256 + j]) * fp(w[col * 256 + j]);
                                }
                                uint16_t gold = bf(fp(bf(sum)) + (biased ? fp(bias[col]) : 0.f));
                                if (a[token * 6144 + col] != gold) {
                                    throw std::runtime_error("CPU projection mismatch");
                                }
                            }
                        }
                    }
                    std::cout << "SHARED_KV_UNIT_PASS rank=" << rank << " variant=" << bad_case << " tokens=" << T << " changed_inputs=5 full_eager_exact=1 CPU_sample_exact=1" << std::endl;
                }
            } catch (std::exception const &e) {
                errors[rank] = e.what();
            }
        });
    }
    for (auto &t : workers) {
        t.join();
    }
    int failed = 0;
    for (int r = 0; r < 4; ++r) {
        if (!errors[r].empty()) {
            std::cerr << "SHARED_KV_UNIT_FAIL rank=" << r << " " << errors[r] << std::endl;
            ++failed;
        }
    }
    return failed;
}
