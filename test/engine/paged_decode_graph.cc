#include "csrc/engine/compiler/paged_compiler.hpp"
#include "csrc/global_state/global_state.hpp"
#include <array>
#include <chrono>
#include <cstring>
#include <infiniccl.h>
#include <infinicore/nn/rope.hpp>
#include <infinicore/ops/distributed/allreduce.hpp>
#include <infinicore/ops/embedding.hpp>
#include <infinicore/ops/mha_varlen.hpp>
#include <infinicore/ops/paged_caching.hpp>
#include <infinicore/ops/select_last_token_hidden.hpp>
#include <iostream>
#include <random>
#include <thread>
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
template <class V>
static Tensor cpu(std::vector<V> const &v, Shape shape, DataType dtype) {
    auto previous = context::getDevice();
    auto t = Tensor::empty(shape, dtype, Device::cpu());
    memcpy(t->data(), v.data(), v.size() * sizeof(V));
    context::setDevice(previous);
    return t;
}

struct Tiny : InfinilmModel {
    Tensor table;
    std::vector<Tensor> caches;
    infinicclComm_t comm;
    Tiny(int rank, infinicclComm_t cm) : comm(cm) {
        auto d = context::getDevice();
        cache_config_ = std::make_unique<cache::PagedKVCacheConfig>(4, 256, 16);
        table = Tensor::empty({64, 2048}, DataType::BF16, d);
        std::vector<uint16_t> h(table->numel());
        for (size_t i = 0; i < h.size(); ++i) {
            h[i] = bf(float((i + rank) % 31) / 32);
        }
        context::memcpyH2D(table->data(), h.data(), h.size() * 2, false);
        for (int l = 0; l < 8; ++l) {
            caches.push_back(Tensor::empty({4, 256, 16, 128}, DataType::BF16, d));
        }
    }
    Output forward(Input const &in) const override {
        size_t T = in.input_ids.value()->numel();
        auto x = op::embedding(in.input_ids.value()->view({T}), table)->view({T, 16, 128});
        for (int l = 0; l < 4; ++l) {
            auto st = Tensor::empty({T, 48, 128}, DataType::BF16, context::getDevice());
            auto k = st->narrow({{1, 16, 16}}), v = st->narrow({{1, 32, 16}});
            k->copy_from(x);
            v->copy_from(x);
            op::paged_caching_(caches[2 * l]->permute({0, 2, 1, 3}), caches[2 * l + 1]->permute({0, 2, 1, 3}), k, v, in.slot_mapping.value());
            auto y = Tensor::empty(x->shape(), DataType::BF16, context::getDevice());
            op::distributed::allreduce_(y, x, INFINICCL_SUM, comm);
            x = y;
        }
        return {x->view({1, T, 2048})};
    }
};
static InfinilmModel::Input input(int B, int phase) {
    InfinilmModel::Input in;
    std::vector<int64_t> ids(B), pos(B), slots(B);
    std::vector<int32_t> offsets(B + 1), lengths(B, 1), blocks(B);
    for (int i = 0; i < B; ++i) {
        ids[i] = (i * 7 + phase) % 64;
        pos[i] = phase;
        slots[i] = phase * 97 + B - 1 - i;
        blocks[i] = slots[i] / 256;
        offsets[i] = i;
    }
    offsets[B] = B;
    in.input_ids = cpu(ids, {1, size_t(B)}, DataType::I64);
    in.position_ids = cpu(pos, {size_t(B)}, DataType::I64);
    in.slot_mapping = cpu(slots, {size_t(B)}, DataType::I64);
    in.block_tables = cpu(blocks, {size_t(B), 1}, DataType::I32);
    in.total_sequence_lengths = cpu(lengths, {size_t(B)}, DataType::I32);
    in.input_offsets = cpu(offsets, {size_t(B + 1)}, DataType::I32);
    in.cu_seqlens = in.input_offsets;
    return in;
}
static InfinilmModel::Input device_input(InfinilmModel::Input in) {
    auto hs = in.slot_mapping.value();
    auto copy = [](std::optional<Tensor> &t) {if(t){t=t.value()->to(context::getDevice());
} };
    copy(in.input_ids);
    copy(in.position_ids);
    copy(in.slot_mapping);
    copy(in.block_tables);
    copy(in.total_sequence_lengths);
    copy(in.input_offsets);
    copy(in.cu_seqlens);
    graph::bind_eager_host_int_array(in.slot_mapping.value(), reinterpret_cast<const int64_t *>(hs->data()), hs->numel());
    return in;
}
extern "C" int check_validated_decode_compiler() {
    std::array<infinicclComm_t, 4> cs{};
    int ids[] = {0, 1, 2, 3};
    if (infinicclCommInitAll(INFINI_DEVICE_ASCEND, cs.data(), 4, ids)) {
        return 1;
    }
    engine::RankBarrier barrier(4);
    std::array<int, 4> error{};
    std::vector<std::thread> workers;
    for (int rank = 0; rank < 4; ++rank) {
        workers.emplace_back([&, rank] {try{Device dev(Device::Type::ASCEND,rank);context::setDevice(dev);auto cfg=std::make_shared<global_state::InfinilmConfig>(backends::AttentionBackend::FLASH_ATTN,nullptr);global_state::initialize_infinilm_config(cfg);global_state::ForwardContext fc;global_state::initialize_forward_context(fc); engine::distributed::RankInfo rank_info(dev); rank_info.tp_size=4; rank_info.tp_rank=rank; rank_info.comm=cs[rank]; global_state::initialize_model_parallel(rank_info);auto model=std::make_shared<Tiny>(rank,cs[rank]);context::setDevice(dev);engine::PagedCompiler compiler(model,&barrier);compiler.compile();
 auto reset=[&]{for(auto cache:model->caches){context::setDeviceMemory(cache->data(),0x55,cache->nbytes());
}context::syncStream();};
 for(int B:{1,4,8,16}){for(int phase=0;phase<5;++phase){graph::clear_eager_host_int_arrays();reset();auto in=input(B,phase);auto[g,out]=compiler.get_compiled(in);if(!g||!out){throw std::runtime_error("decode graph missing");
}barrier.wait();g->run();context::syncStream();std::vector<uint16_t>actual(out->logits->numel());context::memcpyD2H(actual.data(),out->logits->data(),actual.size()*2);std::vector<std::vector<uint16_t>>captured;for(auto c:model->caches){captured.emplace_back(c->numel());context::memcpyD2H(captured.back().data(),c->data(),c->nbytes());}
 reset();auto ref=model->forward(device_input(in));context::syncStream();std::vector<uint16_t>expected(actual.size());context::memcpyD2H(expected.data(),ref.logits->data(),expected.size()*2);if(actual!=expected){throw std::runtime_error("decode graph output differs");
}for(size_t j=0;j<captured.size();++j){std::vector<uint16_t>gold(captured[j].size());context::memcpyD2H(gold.data(),model->caches[j]->data(),gold.size()*2);if(gold!=captured[j]){throw std::runtime_error("decode full cache differs");
}}
 if(B>=4){for(int invalid=0;invalid<6;++invalid){auto bad=input(B,phase);auto*p=reinterpret_cast<int64_t*>(bad.slot_mapping.value()->data());if(invalid==0){p[0]=p[1];
}if(invalid==1){p[0]=-1;
}if(invalid==2){p[0]=1024;
}if(invalid==3){bad.slot_mapping.reset();
}if(invalid==4){bad.slot_mapping=cpu(std::vector<int32_t>(B,1),{size_t(B)},DataType::I32);
}if(invalid==5){bad.slot_mapping=cpu(std::vector<int64_t>(B-1,1),{size_t(B-1)},DataType::I64);
}if(std::get<0>(compiler.get_compiled(bad))){throw std::runtime_error("invalid decode slots accepted");
}}
}
 barrier.wait();std::cout<<"DECODE_COMPILER_PASS rank="<<rank<<" batch="<<B<<" phase="<<phase<<" full_cache_exact=1 invalid_fallback=6"<<std::endl;}
}
 barrier.wait();infinicclCommDestroy(cs[rank]);}catch(std::exception const&e){error[rank]=1;std::cerr<<"DECODE_COMPILER_FAIL rank="<<rank<<" "<<e.what()<<std::endl;} });
    }
    for (auto &t : workers) {
        t.join();
    }
    return error[0] + error[1] + error[2] + error[3];
}
