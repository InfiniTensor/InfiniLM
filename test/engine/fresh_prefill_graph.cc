#include "csrc/engine/compiler/fresh_prefill_compiler.hpp"
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
    mutable Input captured_inputs;
    std::vector<Tensor> caches;
    std::shared_ptr<nn::RoPE> rope;
    infinicclComm_t comm;
    Tiny(int rank, infinicclComm_t cm) : comm(cm) {
        auto d = context::getDevice();
        cache_config_ = std::make_unique<cache::PagedKVCacheConfig>(16, 256, 1);
        table = Tensor::empty({64, 2048}, DataType::BF16, d);
        std::vector<uint16_t> h(table->numel());
        std::mt19937 rng(430 + rank);
        std::uniform_real_distribution<float> u(-.125, .125);
        for (auto &v : h) {
            v = bf(u(rng));
        }
        context::memcpyH2D(table->data(), h.data(), h.size() * 2, false);
        rope = std::make_shared<nn::RoPE>(128, 128, 4096, 1000000., nn::RoPE::Algo::GPT_NEOX, DataType::BF16, d);
        for (int l = 0; l < 8; ++l) {
            caches.push_back(Tensor::empty({16, 256, 16, 128}, DataType::BF16, d));
        }
    }
    Output forward(Input const &in) const override {
        if (context::isGraphRecording()) {
            captured_inputs = in;
        }
        size_t T = in.input_ids.value()->numel();
        auto x = op::embedding(in.input_ids.value()->view({T}), table)->view({T, 16, 128});
        for (int l = 0; l < 4; ++l) {
            auto storage = Tensor::empty({T, 32, 128}, DataType::BF16, context::getDevice());
            auto strided = storage->narrow({{1, 0, 16}});
            strided->copy_from(x);
            x = strided;
            auto qk = rope->forward(x, x, in.position_ids.value()->view({T}));
            op::paged_caching_(caches[2 * l]->permute({0, 2, 1, 3}), caches[2 * l + 1]->permute({0, 2, 1, 3}), qk.second, x, in.slot_mapping.value());
            auto out = Tensor::empty(x->shape(), DataType::BF16, context::getDevice());
            op::mha_varlen_(out, qk.first, qk.second, x, in.input_offsets.value(), in.cu_seqlens.value(), std::nullopt, T, T, std::nullopt, 1 / std::sqrt(128.f));
            op::distributed::allreduce_(out, out, INFINICCL_SUM, comm);
            x = out;
        }
        auto selected = Tensor::empty({1, 1, 2048}, DataType::BF16, context::getDevice());
        op::select_last_token_hidden_(selected, x->view({1, T, 2048}), in.input_offsets.value());
        return {selected};
    }
};
static InfinilmModel::Input make_input(int T, int iteration) {
    InfinilmModel::Input in;
    std::vector<int64_t> ids(T), pos(T), slots(T);
    std::vector<int32_t> blocks((T + 255) / 256);
    int start = iteration * 4;
    for (int i = 0; i < T; ++i) {
        ids[i] = (i * 5 + iteration * 11) % 64;
        pos[i] = (i * 7 + iteration * ((i * i) % 997)) % 4096;
        slots[i] = int64_t((i / 256 + start) % 16) * 256 + i % 256;
    }
    for (size_t i = 0; i < blocks.size(); ++i) {
        blocks[i] = (int(i) + start) % 16;
    }
    in.input_ids = cpu(ids, {1, size_t(T)}, DataType::I64);
    in.position_ids = cpu(pos, {size_t(T)}, DataType::I64);
    in.slot_mapping = cpu(slots, {size_t(T)}, DataType::I64);
    in.block_tables = cpu(blocks, {1, blocks.size()}, DataType::I32);
    in.total_sequence_lengths = cpu(std::vector<int32_t>{T}, {1}, DataType::I32);
    in.past_sequence_lengths = cpu(std::vector<int32_t>{0}, {1}, DataType::I32);
    in.input_offsets = cpu(std::vector<int32_t>{0, T}, {2}, DataType::I32);
    in.cu_seqlens = in.input_offsets;
    return in;
}
static InfinilmModel::Input device_input(InfinilmModel::Input in) {
    auto host_slots = in.slot_mapping.value();
    auto copy = [](std::optional<Tensor> &t) {if(t){t=t.value()->to(context::getDevice());
} };
    copy(in.input_ids);
    copy(in.position_ids);
    copy(in.slot_mapping);
    copy(in.block_tables);
    copy(in.total_sequence_lengths);
    copy(in.past_sequence_lengths);
    copy(in.input_offsets);
    copy(in.cu_seqlens);
    int32_t offs[] = {0, int32_t(in.input_ids.value()->numel())};
    graph::bind_eager_host_int_array(in.input_offsets.value(), offs, 2);
    graph::bind_eager_host_int_array(in.cu_seqlens.value(), offs, 2);
    graph::bind_eager_host_int_array(in.slot_mapping.value(), reinterpret_cast<const int64_t *>(host_slots->data()), host_slots->numel());
    return in;
}
extern "C" int check_fresh_prefill_compiler() {
    std::array<infinicclComm_t, 4> comms{};
    int devices[] = {0, 1, 2, 3};
    if (infinicclCommInitAll(INFINI_DEVICE_ASCEND, comms.data(), 4, devices)) {
        return 1;
    }
    engine::RankBarrier barrier(4);
    std::array<int, 4> errors{};
    std::vector<std::thread> threads;
    for (int rank = 0; rank < 4; ++rank) {
        threads.emplace_back([&, rank] {try{
  Device d(Device::Type::ASCEND,rank);context::setDevice(d);auto config=std::make_shared<global_state::InfinilmConfig>(backends::AttentionBackend::FLASH_ATTN,nullptr);global_state::initialize_infinilm_config(config);global_state::ForwardContext fc;global_state::initialize_forward_context(fc);
  for(int T:{1024,2047,2048}){
   auto model=std::make_shared<Tiny>(rank,comms[rank]);context::setDevice(d);engine::FreshPrefillCompiler compiler(model,&barrier);compiler.compile();float worst=0;
   for(int it=0;it<3;++it){
    if(it>0){
      // Emulate decode reusing the shared arena between two prefills.
      // This catches omitted refreshes of shape-constant device metadata.
      for(auto tensor:{model->captured_inputs.input_ids,model->captured_inputs.position_ids,model->captured_inputs.past_sequence_lengths,model->captured_inputs.total_sequence_lengths,model->captured_inputs.input_offsets,model->captured_inputs.cu_seqlens,model->captured_inputs.block_tables,model->captured_inputs.slot_mapping}){if(tensor){context::setDeviceMemory(tensor.value()->data(),0,tensor.value()->nbytes());
}}
      graph::clear_eager_host_int_arrays();
    }
    auto input=make_input(T,it);auto [g,out]=compiler.get_compiled(input);if(!g||!out){throw std::runtime_error("missing fresh prefill graph");
}barrier.wait();g->run();context::syncStream();std::vector<uint16_t>actual(out->logits->numel());context::memcpyD2H(actual.data(),out->logits->data(),actual.size()*2);
    std::vector<std::vector<uint16_t>> captured_caches;for(auto cache:model->caches){captured_caches.emplace_back(cache->numel());context::memcpyD2H(captured_caches.back().data(),cache->data(),cache->numel()*2);}
    barrier.wait();auto eager_input=device_input(input);auto expected=model->forward(eager_input);context::syncStream();std::vector<uint16_t>reference(actual.size());context::memcpyD2H(reference.data(),expected.logits->data(),reference.size()*2);for(size_t i=0;i<actual.size();++i){float a=fp(actual[i]),b=fp(reference[i]),e=std::abs(a-b);worst=std::max(worst,e);if(!std::isfinite(a)||e>.01f+.01f*std::abs(b)){throw std::runtime_error("fresh prefill graph vs eager mismatch");
}}
    for(size_t ci=0;ci<model->caches.size();++ci){auto cache=model->caches[ci];std::vector<uint16_t>gold(cache->numel());context::memcpyD2H(gold.data(),cache->data(),gold.size()*2);if(gold!=captured_caches[ci]){throw std::runtime_error("cache location/value mismatch after graph replay");
}}
    // A prefix hit, duplicate slots, changed shape and speculative output must
    // all refuse the existing entry, rather than reuse captured metadata.
    auto bad=input;bad.past_sequence_lengths=cpu(std::vector<int32_t>{1},{1},DataType::I32);if(std::get<0>(compiler.get_compiled(bad))){throw std::runtime_error("prefix accepted");
}
    bad=input;bad.sample_all_positions=true;if(std::get<0>(compiler.get_compiled(bad))){throw std::runtime_error("speculative accepted");
}
    bad=make_input(T==1024?2048:1024,it);if(std::get<0>(compiler.get_compiled(bad))){throw std::runtime_error("new shape accepted into bounded cache");
}
    bad=input;std::vector<int64_t>dup(T,0);bad.slot_mapping=cpu(dup,{size_t(T)},DataType::I64);if(std::get<0>(compiler.get_compiled(bad))){throw std::runtime_error("duplicate slots accepted");
}
    barrier.wait();
   }
   std::cout<<"PASS FRESH_PREFILL_TP4_FOUR_LAYER rank="<<rank<<" tokens="<<T<<" changed_ids_positions_slots=3 metadata_poison_recovery=2 guards=12 max_abs="<<worst<<std::endl;compiler.compile();context::syncStream();barrier.wait();
  }
  infinicclCommDestroy(comms[rank]);
 }catch(std::exception const&e){errors[rank]=1;std::cerr<<"FAIL rank="<<rank<<" "<<e.what()<<std::endl;} });
    }
    for (auto &t : threads) {
        t.join();
    }
    return errors[0] + errors[1] + errors[2] + errors[3];
}
