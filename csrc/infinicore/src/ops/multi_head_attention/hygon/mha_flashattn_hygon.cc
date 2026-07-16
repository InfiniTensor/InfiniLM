#if defined(ENABLE_HYGON_API) && defined(ENABLE_FLASH_ATTN)
#include "infinicore/ops/mha.hpp"

#include "../../../adaptor/flash_attn/hygon/flash_attn_hygon.hpp"
#include "infinicore/adaptor/aten_adaptor.hpp"

#include <limits>
#include <stdexcept>

#include <c10/hip/HIPGuard.h>

namespace infinicore::op::mha_impl::flashattn {
struct PlannedMeta {
    graph::GraphTensor out, q, k, v;
    std::optional<graph::GraphTensor> alibi_slopes;
    float scale;
    bool is_causal;
};

void *plan(Tensor out,
           const Tensor &q,
           const Tensor &k,
           const Tensor &v,
           std::optional<Tensor> alibi_slopes,
           float scale,
           bool is_causal) {

    return new PlannedMeta{
        graph::GraphTensor(out),
        graph::GraphTensor(q),
        graph::GraphTensor(k),
        graph::GraphTensor(v),
        alibi_slopes ? std::optional<graph::GraphTensor>(graph::GraphTensor(*alibi_slopes)) : std::nullopt,
        scale,
        is_causal};
}

void run(void *planned_meta) {
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);

    auto q = infinicore::adaptor::to_aten_tensor(p->q);
    auto k = infinicore::adaptor::to_aten_tensor(p->k);
    auto v = infinicore::adaptor::to_aten_tensor(p->v);

    const bool out_need_copy_back = !p->out->is_contiguous();
    Tensor out_work_ic = out_need_copy_back ? p->out->contiguous() : Tensor(p->out);
    auto out_work = infinicore::adaptor::to_aten_tensor(out_work_ic);
    auto scale = p->scale;
    auto is_causal = p->is_causal;

    const int64_t num_q_heads = q.size(2);
    const int64_t num_kv_heads = k.size(2);
    if (q.dim() == 4 && k.dim() == 4 && v.dim() == 4 && q.size(1) == 1 && k.size(1) == 1 && v.size(1) == 1) {
        at::Tensor value = v;
        if (num_kv_heads != num_q_heads) {
            const int64_t repeat = num_q_heads / num_kv_heads;
            value = v.unsqueeze(3).repeat({1, 1, 1, repeat, 1}).reshape(q.sizes());
        }
        out_work.copy_(value);
        if (out_need_copy_back) {
            p->out->copy_from(out_work_ic);
        }
        return;
    }

    at::Tensor k_attn = k;
    at::Tensor v_attn = v;
    if (num_kv_heads != num_q_heads) {
        const int64_t repeat = num_q_heads / num_kv_heads;
        k_attn = k.unsqueeze(3).repeat({1, 1, 1, repeat, 1}).reshape(q.sizes());
        v_attn = v.unsqueeze(3).repeat({1, 1, 1, repeat, 1}).reshape(q.sizes());
    }

    auto attn_weight = at::matmul(q.permute({0, 2, 1, 3}), k_attn.permute({0, 2, 3, 1})) * scale;
    if (is_causal) {
        auto mask = at::tril(at::ones_like(attn_weight), -1).flip({-2, -1});
        auto neg_inf = at::full_like(attn_weight, -std::numeric_limits<float>::infinity());
        attn_weight = at::where(mask == 1, neg_inf, attn_weight);
    }
    auto attn = at::softmax(attn_weight, -1);
    auto result = at::matmul(attn, v_attn.permute({0, 2, 1, 3})).permute({0, 2, 1, 3});
    out_work.copy_(result);
    if (out_need_copy_back) {
        p->out->copy_from(out_work_ic);
    }
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    MultiheadAttention::plan_dispatcher().registerDevice(Device::Type::kHygon, &plan);
    MultiheadAttention::run_dispatcher().registerDevice(Device::Type::kHygon, &run);
    MultiheadAttention::cleanup_dispatcher().registerDevice(Device::Type::kHygon, &cleanup);
    return true;
}();

} // namespace infinicore::op::mha_impl::flashattn
#endif // ENABLE_HYGON_API && ENABLE_FLASH_ATTN
