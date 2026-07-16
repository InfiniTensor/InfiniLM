#if defined(ENABLE_MOORE_MATE_FLASH_ATTN)

#include "infinicore/ops/mha_kvcache.hpp"

#include "infinicore/adaptor/aten_adaptor.hpp"

#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <torch/csrc/utils/pybind.h>

namespace infinicore::op::mha_kvcache_impl::flashattn_moore {

namespace py = pybind11;

// Lightweight RAII: Binds MUSA streams,
// avoiding the need to include <c10/musa/MUSAGuard.h>
namespace {
class LocalMUSAStreamGuard {
public:
    explicit LocalMUSAStreamGuard(const c10::musa::MUSAStream &s)
        : prev_(c10::musa::getCurrentMUSAStream(s.device_index())) {
        c10::musa::setCurrentMUSAStream(s);
    }
    ~LocalMUSAStreamGuard() {
        c10::musa::setCurrentMUSAStream(prev_);
    }
    LocalMUSAStreamGuard(const LocalMUSAStreamGuard &) = delete;
    LocalMUSAStreamGuard &operator=(const LocalMUSAStreamGuard &) = delete;

private:
    c10::musa::MUSAStream prev_;
};
} // namespace

struct PlannedMeta {
    graph::GraphTensor out, q, k_cache, v_cache, seqlens_k, block_table;
    std::optional<graph::GraphTensor> alibi_slopes;
    float scale;
};

void *plan(Tensor out,
           const Tensor &q,
           const Tensor &k_cache,
           const Tensor &v_cache,
           const Tensor &seqlens_k,
           const Tensor &block_table,
           std::optional<Tensor> alibi_slopes,
           float scale) {
    return new PlannedMeta{
        graph::GraphTensor(out),
        graph::GraphTensor(q),
        graph::GraphTensor(k_cache),
        graph::GraphTensor(v_cache),
        graph::GraphTensor(seqlens_k),
        graph::GraphTensor(block_table),
        alibi_slopes ? std::optional<graph::GraphTensor>(graph::GraphTensor(*alibi_slopes)) : std::nullopt,
        scale};
}

void run(void *planned_meta) {
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);
    if (p->alibi_slopes.has_value()) {
        throw std::runtime_error(
            "[mha_kvcache/moore] ALiBi not supported by mate flash_attn_with_kvcache");
    }

    LocalMUSAStreamGuard guard(infinicore::adaptor::get_musa_stream());

    auto out_tensor = infinicore::adaptor::to_aten_tensor(p->out);
    auto q_4d = infinicore::adaptor::to_aten_tensor(p->q);
    auto k_cache = infinicore::adaptor::to_aten_tensor(p->k_cache);
    auto v_cache = infinicore::adaptor::to_aten_tensor(p->v_cache);
    auto seqlens_k = infinicore::adaptor::to_aten_tensor(p->seqlens_k);
    auto block_table = infinicore::adaptor::to_aten_tensor(p->block_table);

    auto q_3d = q_4d.squeeze(1);

    const int64_t block_size = k_cache.size(1);
    const int64_t max_seq_len = block_table.size(1) * block_size;

    try {
        py::gil_scoped_acquire gil;
        py::module_ wrapper = py::module_::import("infinicore.ops.moore_mate_flash_attn");

        py::object py_q = py::cast(q_3d);
        py::object py_k_cache = py::cast(k_cache);
        py::object py_v_cache = py::cast(v_cache);
        py::object py_seqlens_k = py::cast(seqlens_k);
        py::object py_blk_tbl = py::cast(block_table);

        py::object result = wrapper.attr("moore_mate_flash_attn_decode")(
            py_q,
            py_k_cache,
            py_v_cache,
            py_blk_tbl,
            py_seqlens_k,
            p->scale,
            block_size,
            max_seq_len);

        at::Tensor result_t = result.cast<at::Tensor>();
        out_tensor.copy_(result_t.unsqueeze(1));

        result = py::none();
        py_q = py_k_cache = py_v_cache = py_seqlens_k = py_blk_tbl = py::none();
    } catch (const py::error_already_set &e) {
        throw std::runtime_error(
            std::string("[mha_kvcache/moore] Python error: ") + e.what());
    }
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    MhaKVCache::plan_dispatcher().registerDevice(Device::Type::kMoore, &plan);
    MhaKVCache::run_dispatcher().registerDevice(Device::Type::kMoore, &run);
    MhaKVCache::cleanup_dispatcher().registerDevice(Device::Type::kMoore, &cleanup);
    return true;
}();

} // namespace infinicore::op::mha_kvcache_impl::flashattn_moore

#endif // ENABLE_MOORE_MATE_FLASH_ATTN
