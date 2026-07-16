#if defined(ENABLE_MOORE_MATE_FLASH_ATTN)

#include "infinicore/ops/mha_varlen.hpp"

#include "infinicore/adaptor/aten_adaptor.hpp"

#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <torch/csrc/utils/pybind.h>

namespace infinicore::op::mha_varlen_impl::flashattn_moore {

namespace py = pybind11;

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
    graph::GraphTensor out, q, k, v, cum_seqlens_q, cum_seqlens_k;
    std::optional<graph::GraphTensor> block_table;
    int max_seqlen_q, max_seqlen_k;
    std::optional<graph::GraphTensor> alibi_slopes;
    float scale;
};

void *plan(Tensor out,
           const Tensor &q,
           const Tensor &k,
           const Tensor &v,
           const Tensor &cum_seqlens_q,
           const Tensor &cum_seqlens_k,
           std::optional<Tensor> block_table,
           int max_seqlen_q,
           int max_seqlen_k,
           std::optional<Tensor> alibi_slopes,
           float scale) {
    return new PlannedMeta{
        graph::GraphTensor(out),
        graph::GraphTensor(q),
        graph::GraphTensor(k),
        graph::GraphTensor(v),
        graph::GraphTensor(cum_seqlens_q),
        graph::GraphTensor(cum_seqlens_k),
        block_table ? std::optional<graph::GraphTensor>(graph::GraphTensor(*block_table)) : std::nullopt,
        max_seqlen_q,
        max_seqlen_k,
        alibi_slopes ? std::optional<graph::GraphTensor>(graph::GraphTensor(*alibi_slopes)) : std::nullopt,
        scale};
}

void run(void *planned_meta) {
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);
    if (p->alibi_slopes.has_value()) {
        throw std::runtime_error(
            "[mha_varlen/moore] ALiBi not supported by mate flash_attn_varlen");
    }
    if (!p->block_table.has_value()) {
        throw std::runtime_error(
            "[mha_varlen/moore] dense KV is not supported by mate flash_attn_varlen");
    }

    LocalMUSAStreamGuard guard(infinicore::adaptor::get_musa_stream());

    auto out_tensor = infinicore::adaptor::to_aten_tensor(p->out);
    auto q_tensor = infinicore::adaptor::to_aten_tensor(p->q);
    auto k_cache = infinicore::adaptor::to_aten_tensor(p->k);
    auto v_cache = infinicore::adaptor::to_aten_tensor(p->v);
    auto cu_seqlens_q = infinicore::adaptor::to_aten_tensor(p->cum_seqlens_q);
    auto cu_seqlens_k = infinicore::adaptor::to_aten_tensor(p->cum_seqlens_k);
    auto block_table = infinicore::adaptor::to_aten_tensor(*p->block_table);

    const int64_t block_size = k_cache.size(1);

    int max_seqlen_q_bound = static_cast<int>(q_tensor.size(0));
    int max_seqlen_k_bound = static_cast<int>(q_tensor.size(0));

    try {
        py::gil_scoped_acquire gil;
        py::module_ wrapper = py::module_::import("infinicore.ops.moore_mate_flash_attn");

        py::object py_q = py::cast(q_tensor);
        py::object py_k = py::cast(k_cache);
        py::object py_v = py::cast(v_cache);
        py::object py_cuq = py::cast(cu_seqlens_q);
        py::object py_cuk = py::cast(cu_seqlens_k);
        py::object py_blk = py::cast(block_table);

        py::object result = wrapper.attr("moore_mate_flash_attn_prefill")(
            py_q,
            py_k,
            py_v,
            py_cuq,
            py_cuk,
            py_blk,
            p->scale,
            max_seqlen_q_bound,
            max_seqlen_k_bound,
            block_size,
            true);

        at::Tensor result_t = result.cast<at::Tensor>();
        out_tensor.copy_(result_t);
    } catch (const py::error_already_set &e) {
        throw std::runtime_error(
            std::string("[mha_varlen/moore] Python error: ") + e.what());
    }
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    MultiheadAttentionVarlen::plan_dispatcher().registerDevice(Device::Type::kMoore, &plan);
    MultiheadAttentionVarlen::run_dispatcher().registerDevice(Device::Type::kMoore, &run);
    MultiheadAttentionVarlen::cleanup_dispatcher().registerDevice(Device::Type::kMoore, &cleanup);
    return true;
}();

} // namespace infinicore::op::mha_varlen_impl::flashattn_moore

#endif // ENABLE_MOORE_MATE_FLASH_ATTN
