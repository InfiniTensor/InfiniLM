// Hygon DCU flash-attention via dlsym from the system flash_attn_2_cuda*.so
// (DAS/DTK fork). Compiled as plain host C++ (NOT hipcc/nvcc) so it can pull
// <c10/hip/...> headers without conflicting with the rest of libinfinicore
// (which compiles via DTK's CUDA-compat nvcc shim and would re-define HIP
// types). __HIP_PLATFORM_AMD__ is set in CMakeLists.
//
// Mirrors `InfiniTensor/InfiniCore` branch `hygon-flash-attn`:
//   src/infinicore/adaptor/flash_attn_hygon_wrapper.cc           (dlsym fns)
//   src/infinicore/ops/mha_kvcache/mha_kvcache_hygon_paged.cc    (decode op)
//   src/infinicore/ops/multi_head_attention_varlen/mha_varlen_hygon_vllm.cc
//                                                                (prefill op)
// consolidated into a single TU here.
//
// Why dlsym instead of link-time:
//   - The DTK flash-attn wheel ships a Python extension .so. Linking against
//     it would pull in libtorch_python (and break a pure C++ host).
//
// Why HIPStreamGuard at the top of run():
//   - DTK's ATen runs ops on PyTorch's "current" stream by default. To get the
//     kernels onto InfiniLM's stream we must bind it via HIPStreamGuard before
//     any ATen call (session_export.md §3 Bug B2).
//
// Why paged_attention for decode (not mha_fwd_kvcache):
//   - mha_fwd_kvcache calls torch::empty for softmax_lse on every step which
//     breaks HIP graph capture. paged_attention uses only static allocations.
//
// Symbol signature reference (DTK fork, c10::optional throughout):
//   mha_fwd_kvcache(q, kcache, vcache, k_, v_, seqlens_k_, rotary_cos_,
//                   rotary_sin_, cache_batch_idx_, leftpad_k_, block_table_,
//                   alibi_slopes_, out_, softmax_scale, is_causal,
//                   window_size_left, window_size_right, softcap,
//                   is_rotary_interleaved, num_splits, s_aux_)
//   vllm_mha_varlen_fwd(q, k, v, out_, cu_seqlens_q, cu_seqlens_k, seqused_k,
//                       leftpad_k_, block_table_, alibi_slopes_, max_seqlen_q,
//                       max_seqlen_k, p_dropout, softmax_scale, zero_tensors,
//                       is_causal, window_size_left, window_size_right,
//                       softcap, return_softmax, q_descale_, k_descale_,
//                       v_descale_, gen_, s_aux_)
//   paged_attention(out, q, k_cache, v_cache, scale, block_table, context_lens,
//                   alibi_slopes, kv_cache_dtype, q_scale, k_scale, v_scale,
//                   max_context_len, s_aux)

#if defined(ENABLE_HYGON_API) && defined(ENABLE_FLASH_ATTN_DLSYM)

#include "infinicore/context/context.hpp"
#include "infinicore/graph/graph.hpp"
#include "infinicore/ops/common/dispatcher.hpp"
#include "infinicore/ops/mha_kvcache.hpp"
#include "infinicore/ops/mha_varlen.hpp"
#include "infinicore/tensor.hpp"

#include <ATen/ATen.h>
#include <c10/hip/HIPGuard.h>
#include <c10/hip/HIPStream.h>
#include <c10/util/Optional.h>

#include <dlfcn.h>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <hip/hip_runtime_api.h>

// =====================================================================
// dlsym function-pointer types — match DTK's exact `extern "C"` signatures.
// DTK uses c10::optional throughout; `c10::optional` is `using std::optional`
// in modern PyTorch but the type identity matters for ABI matching.
// =====================================================================

namespace infinicore::adaptor::hygon_fa {

namespace {

using mha_fwd_kvcache_fn_t = std::vector<at::Tensor> (*)(
    at::Tensor & /*q*/,
    const at::Tensor & /*kcache*/,
    const at::Tensor & /*vcache*/,
    c10::optional<const at::Tensor> & /*k_*/,
    c10::optional<const at::Tensor> & /*v_*/,
    c10::optional<const at::Tensor> & /*seqlens_k_*/,
    c10::optional<const at::Tensor> & /*rotary_cos_*/,
    c10::optional<const at::Tensor> & /*rotary_sin_*/,
    c10::optional<const at::Tensor> & /*cache_batch_idx_*/,
    c10::optional<const at::Tensor> & /*leftpad_k_*/,
    c10::optional<at::Tensor> & /*block_table_*/,
    c10::optional<at::Tensor> & /*alibi_slopes_*/,
    c10::optional<at::Tensor> & /*out_*/,
    const float /*softmax_scale*/,
    bool /*is_causal*/,
    int /*window_size_left*/,
    int /*window_size_right*/,
    const float /*softcap*/,
    bool /*is_rotary_interleaved*/,
    int /*num_splits*/,
    const c10::optional<at::Tensor> & /*s_aux_*/);

using mha_varlen_fwd_fn_t = std::vector<at::Tensor> (*)(
    at::Tensor & /*q*/,
    const at::Tensor & /*k*/,
    const at::Tensor & /*v*/,
    c10::optional<at::Tensor> & /*out_*/,
    const at::Tensor & /*cu_seqlens_q*/,
    const at::Tensor & /*cu_seqlens_k*/,
    c10::optional<at::Tensor> & /*seqused_k*/,
    c10::optional<const at::Tensor> & /*leftpad_k_*/,
    c10::optional<at::Tensor> & /*block_table_*/,
    c10::optional<at::Tensor> & /*alibi_slopes_*/,
    int /*max_seqlen_q*/,
    const int /*max_seqlen_k*/,
    const float /*p_dropout*/,
    const float /*softmax_scale*/,
    const bool /*zero_tensors*/,
    bool /*is_causal*/,
    int /*window_size_left*/,
    int /*window_size_right*/,
    const float /*softcap*/,
    const bool /*return_softmax*/,
    c10::optional<at::Tensor> /*q_descale_*/,
    c10::optional<at::Tensor> /*k_descale_*/,
    c10::optional<at::Tensor> /*v_descale_*/,
    c10::optional<at::Generator> /*gen_*/,
    const c10::optional<at::Tensor> & /*s_aux_*/);

using paged_attention_fn_t = void (*)(
    at::Tensor & /*out*/,
    at::Tensor & /*q*/,
    at::Tensor & /*k_cache*/,
    at::Tensor & /*v_cache*/,
    double /*scale*/,
    at::Tensor & /*block_table*/,
    at::Tensor & /*context_lens*/,
    const c10::optional<at::Tensor> & /*alibi_slopes*/,
    const std::string & /*kv_cache_dtype*/,
    const c10::optional<at::Tensor> & /*q_scale*/,
    const c10::optional<at::Tensor> & /*k_scale*/,
    const c10::optional<at::Tensor> & /*v_scale*/,
    int /*max_context_len*/,
    const c10::optional<at::Tensor> & /*s_aux*/);

mha_fwd_kvcache_fn_t s_mha_fwd_kvcache  = nullptr;
mha_varlen_fwd_fn_t  s_mha_varlen_fwd   = nullptr;
mha_varlen_fwd_fn_t  s_vllm_varlen_fwd  = nullptr;
paged_attention_fn_t s_paged_attention  = nullptr;

void *resolve(const char *name) {
    void *p = dlsym(RTLD_DEFAULT, name);
    if (p) return p;
    // Fall back: dlopen the wheel (python/infinilm/__init__.py preloads it).
    void *h = dlopen("flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so",
                     RTLD_LAZY | RTLD_GLOBAL);
    if (!h) h = dlopen("flash_attn_2_cuda.so", RTLD_LAZY | RTLD_GLOBAL);
    if (!h) {
        throw std::runtime_error(
            std::string("Hygon flash-attn dlsym: flash_attn_2_cuda*.so not loaded. ") +
            "Either `import flash_attn` before InfiniLM (see " +
            "python/infinilm/__init__.py RTLD_GLOBAL preload) or set LD_PRELOAD.");
    }
    p = dlsym(h, name);
    if (!p) {
        throw std::runtime_error(
            std::string("Hygon flash-attn dlsym: symbol '") + name +
            "' not found in flash_attn_2_cuda*.so");
    }
    return p;
}

void resolve_once() {
    static std::once_flag flag;
    std::call_once(flag, []() {
        s_mha_fwd_kvcache = reinterpret_cast<mha_fwd_kvcache_fn_t>(resolve("mha_fwd_kvcache"));
        s_mha_varlen_fwd  = reinterpret_cast<mha_varlen_fwd_fn_t>(resolve("mha_varlen_fwd"));
        s_vllm_varlen_fwd = reinterpret_cast<mha_varlen_fwd_fn_t>(resolve("vllm_mha_varlen_fwd"));
        s_paged_attention = reinterpret_cast<paged_attention_fn_t>(resolve("paged_attention"));
    });
}

at::Tensor make_aten_tensor(const infinicore::Tensor &t) {
    void *data_ptr = const_cast<void *>(reinterpret_cast<const void *>(t->data()));
    auto sizes = std::vector<int64_t>(t->shape().begin(), t->shape().end());
    auto strides = t->strides();
    at::ScalarType dtype;
    switch (t->dtype()) {
    case infinicore::DataType::F32:  dtype = at::kFloat;    break;
    case infinicore::DataType::F16:  dtype = at::kHalf;     break;
    case infinicore::DataType::BF16: dtype = at::kBFloat16; break;
    case infinicore::DataType::I32:  dtype = at::kInt;      break;
    case infinicore::DataType::I64:  dtype = at::kLong;     break;
    default:
        throw std::runtime_error("hygon_fa: unsupported dtype for ATen wrapper");
    }
    auto deleter = [](void *) {};
    auto opts = at::TensorOptions()
                    .dtype(dtype)
                    .device(at::Device(at::kCUDA, t->device().getIndex()))
                    .requires_grad(false);
    return at::from_blob(data_ptr, sizes, strides, deleter, opts);
}

c10::hip::HIPStream get_hip_stream_from_context() {
    auto *raw = static_cast<hipStream_t>(infinicore::context::getStream());
    return c10::hip::getStreamFromExternal(raw, infinicore::context::getDevice().getIndex());
}

} // namespace

// =====================================================================
// MhaKVCache (decode path) — uses paged_attention.
// InfiniLM passes K/V in BSHD layout [num_blocks, block_size, nkv, hd].
// paged_attention expects:
//   K: [num_blocks, nkv, block_size, hd]
//   V: [num_blocks, nkv, hd, block_size]   (transposed)
// =====================================================================

namespace mha_kvcache {

struct PlannedMeta {
    infinicore::graph::GraphTensor out, q, k_cache, v_cache, seqlens_k, block_table;
    std::optional<infinicore::graph::GraphTensor> alibi_slopes;
    float scale;
    // Pre-allocated scratches (allocated in plan(), reused across runs):
    //   v_transposed_buf:  [num_blocks, nkv, hd, block_size] for paged_attention V layout
    //   k_permuted_buf:    [num_blocks, nkv, block_size, hd] for paged_attention K layout
    // Pre-allocating both keeps destination addresses stable across HIP graph
    // replays and matches the working InfiniOps implementation.
    infinicore::graph::GraphTensor v_transposed_buf;
    infinicore::graph::GraphTensor k_permuted_buf;
};

void *plan(infinicore::Tensor out,
           const infinicore::Tensor &q,
           const infinicore::Tensor &k_cache,
           const infinicore::Tensor &v_cache,
           const infinicore::Tensor &seqlens_k,
           const infinicore::Tensor &block_table,
           std::optional<infinicore::Tensor> alibi_slopes,
           float scale) {
    // K/V cache shape from InfiniLM: {num_blocks, block_size, nkv, hd}.
    auto ks = k_cache->shape();
    auto vs = v_cache->shape();
    // K-permuted: {num_blocks, nkv, block_size, hd}
    auto k_permuted = infinicore::Tensor::empty(
        {ks[0], ks[2], ks[1], ks[3]},
        k_cache->dtype(),
        k_cache->device());
    // V-transposed: {num_blocks, nkv, hd, block_size}
    auto v_transposed = infinicore::Tensor::empty(
        {vs[0], vs[2], vs[3], vs[1]},
        v_cache->dtype(),
        v_cache->device());

    return new PlannedMeta{
        infinicore::graph::GraphTensor(out),
        infinicore::graph::GraphTensor(q),
        infinicore::graph::GraphTensor(k_cache),
        infinicore::graph::GraphTensor(v_cache),
        infinicore::graph::GraphTensor(seqlens_k),
        infinicore::graph::GraphTensor(block_table),
        alibi_slopes ? std::optional<infinicore::graph::GraphTensor>(infinicore::graph::GraphTensor(*alibi_slopes)) : std::nullopt,
        scale,
        infinicore::graph::GraphTensor(v_transposed),
        infinicore::graph::GraphTensor(k_permuted)};
}

void run(void *planned_meta) {
    resolve_once();
    c10::hip::HIPStreamGuard guard(get_hip_stream_from_context());

    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);

    auto out_t = make_aten_tensor(p->out);
    auto q_t   = make_aten_tensor(p->q).contiguous();
    auto k_raw = make_aten_tensor(p->k_cache);
    auto v_raw = make_aten_tensor(p->v_cache);
    auto k_t   = make_aten_tensor(p->k_permuted_buf);
    auto v_t   = make_aten_tensor(p->v_transposed_buf);
    auto seqlens   = make_aten_tensor(p->seqlens_k);
    auto block_tbl = make_aten_tensor(p->block_table);

    // K permute into pre-allocated scratch: [num_blocks, BS, nkv, hd] → [num_blocks, nkv, BS, hd]
    k_t.copy_(k_raw.permute({0, 2, 1, 3}));
    // V transpose into pre-allocated scratch: [num_blocks, BS, nkv, hd] → [num_blocks, nkv, hd, BS]
    v_t.copy_(v_raw.permute({0, 2, 3, 1}));

    // max_context_len upper-bound from tensor shapes (avoids D2H sync that
    // would break HIP graph capture).
    int block_size         = static_cast<int>(k_t.size(2));
    int max_blocks_per_seq = static_cast<int>(block_tbl.size(1));
    int max_context_len    = max_blocks_per_seq * block_size;

    c10::optional<at::Tensor> alibi    = c10::nullopt;
    c10::optional<at::Tensor> q_scale  = c10::nullopt;
    c10::optional<at::Tensor> k_scale  = c10::nullopt;
    c10::optional<at::Tensor> v_scale  = c10::nullopt;
    c10::optional<at::Tensor> s_aux    = c10::nullopt;

    s_paged_attention(
        out_t,            // [num_seqs, 1, num_heads, hd]
        q_t,              // [num_seqs, 1, num_heads, hd]
        k_t,              // [num_blocks, nkv, block_size, hd]
        v_t,              // [num_blocks, nkv, hd, block_size]
        static_cast<double>(p->scale),
        block_tbl,
        seqlens,
        alibi,
        std::string("auto"),
        q_scale, k_scale, v_scale,
        max_context_len,
        s_aux);
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace mha_kvcache

// =====================================================================
// MultiheadAttentionVarlen (prefill path) — uses vllm_mha_varlen_fwd.
// =====================================================================

namespace mha_varlen {

struct PlannedMeta {
    infinicore::graph::GraphTensor out, q, k, v, cum_seqlens_q, cum_seqlens_k, block_table;
    int max_seqlen_q, max_seqlen_k;
    std::optional<infinicore::graph::GraphTensor> alibi_slopes;
    float scale;
};

void *plan(infinicore::Tensor out,
           const infinicore::Tensor &q,
           const infinicore::Tensor &k,
           const infinicore::Tensor &v,
           const infinicore::Tensor &cum_seqlens_q,
           const infinicore::Tensor &cum_seqlens_k,
           const infinicore::Tensor &block_table,
           int max_seqlen_q,
           int max_seqlen_k,
           std::optional<infinicore::Tensor> alibi_slopes,
           float scale) {
    return new PlannedMeta{
        infinicore::graph::GraphTensor(out),
        infinicore::graph::GraphTensor(q),
        infinicore::graph::GraphTensor(k),
        infinicore::graph::GraphTensor(v),
        infinicore::graph::GraphTensor(cum_seqlens_q),
        infinicore::graph::GraphTensor(cum_seqlens_k),
        infinicore::graph::GraphTensor(block_table),
        max_seqlen_q,
        max_seqlen_k,
        alibi_slopes ? std::optional<infinicore::graph::GraphTensor>(infinicore::graph::GraphTensor(*alibi_slopes)) : std::nullopt,
        scale};
}

void run(void *planned_meta) {
    resolve_once();
    c10::hip::HIPStreamGuard guard(get_hip_stream_from_context());

    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);

    // q can come in non-contiguous (slice of a fused QKV projection); flash-attn
    // requires contiguous q.
    auto q_t = make_aten_tensor(p->q).contiguous();
    // K: [num_blocks, block_size, nkv, hd] → [num_blocks, nkv, block_size, hd]
    auto k_t = make_aten_tensor(p->k).permute({0, 2, 1, 3}).contiguous();
    // V: [num_blocks, block_size, nkv, hd] → [num_blocks, nkv, hd, block_size] (transposed)
    auto v_t = make_aten_tensor(p->v).permute({0, 2, 3, 1}).contiguous();
    c10::optional<at::Tensor> out_opt = make_aten_tensor(p->out).contiguous();
    auto cu_seqlens_q  = make_aten_tensor(p->cum_seqlens_q);
    auto cu_seqlens_kv = make_aten_tensor(p->cum_seqlens_k);
    c10::optional<at::Tensor> block_table_opt = make_aten_tensor(p->block_table);

    auto device = q_t.device();
    if (!cu_seqlens_q.is_cuda())  cu_seqlens_q  = cu_seqlens_q.to(device);
    if (!cu_seqlens_kv.is_cuda()) cu_seqlens_kv = cu_seqlens_kv.to(device);
    if (block_table_opt.has_value() && !block_table_opt->is_cuda()) {
        block_table_opt = block_table_opt->to(device);
    }

    c10::optional<at::Tensor>       seqused_k = c10::nullopt;
    c10::optional<const at::Tensor> leftpad_k = c10::nullopt;
    c10::optional<at::Tensor>       alibi     = p->alibi_slopes
                                                    ? c10::optional<at::Tensor>(make_aten_tensor(*p->alibi_slopes))
                                                    : c10::nullopt;
    c10::optional<at::Tensor>       q_descale = c10::nullopt;
    c10::optional<at::Tensor>       k_descale = c10::nullopt;
    c10::optional<at::Tensor>       v_descale = c10::nullopt;
    c10::optional<at::Generator>    gen       = c10::nullopt;
    c10::optional<at::Tensor>       s_aux     = c10::nullopt;

    // Caller passes max_position_embeddings_ (e.g. 65536) as both max_seqlens,
    // but DTK FA uses these for internal workspace sizing — oversized values
    // VMFault. Use tensor-shape upper bounds (per InfiniOps reference) —
    // tightens to actual prefill total tokens + paged k extent.
    const int max_seqlen_q = static_cast<int>(q_t.size(0));
    const int max_seqlen_k = block_table_opt.has_value()
                                 ? static_cast<int>(block_table_opt->size(1) * k_t.size(2))
                                 : max_seqlen_q;

    (void)s_vllm_varlen_fwd(
        q_t, k_t, v_t, out_opt,
        cu_seqlens_q, cu_seqlens_kv,
        seqused_k, leftpad_k, block_table_opt, alibi,
        max_seqlen_q, max_seqlen_k,
        /*p_dropout*/ 0.0f,
        p->scale,
        /*zero_tensors*/ false,
        /*is_causal*/ true,
        /*window_size_left*/ -1,
        /*window_size_right*/ -1,
        /*softcap*/ 0.0f,
        /*return_softmax*/ false,
        q_descale, k_descale, v_descale,
        gen,
        s_aux);
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace mha_varlen

} // namespace infinicore::adaptor::hygon_fa

// Lazy-registration: libinfinicore calls this from MhaKVCache /
// MultiheadAttentionVarlen ctors (cannot be a static initializer here because
// libflash_attn_hygon_dlsym is loaded *during* libinfinicore's relocation, so
// MhaKVCache::*_dispatcher() symbols aren't yet resolved).
//
// Register only for HYGON, with override_existing=true, so we replace the
// generic flashattn registrations from the codebase's existing TUs.
extern "C" void infinicore_hygon_register_flash_attn() {
    static std::once_flag flag;
    std::call_once(flag, []() {
        infinicore::op::MhaKVCache::plan_dispatcher().registerDevice(
            infinicore::Device::Type::HYGON,
            &infinicore::adaptor::hygon_fa::mha_kvcache::plan, true);
        infinicore::op::MhaKVCache::run_dispatcher().registerDevice(
            infinicore::Device::Type::HYGON,
            &infinicore::adaptor::hygon_fa::mha_kvcache::run, true);
        infinicore::op::MhaKVCache::cleanup_dispatcher().registerDevice(
            infinicore::Device::Type::HYGON,
            &infinicore::adaptor::hygon_fa::mha_kvcache::cleanup, true);

        infinicore::op::MultiheadAttentionVarlen::plan_dispatcher().registerDevice(
            infinicore::Device::Type::HYGON,
            &infinicore::adaptor::hygon_fa::mha_varlen::plan, true);
        infinicore::op::MultiheadAttentionVarlen::run_dispatcher().registerDevice(
            infinicore::Device::Type::HYGON,
            &infinicore::adaptor::hygon_fa::mha_varlen::run, true);
        infinicore::op::MultiheadAttentionVarlen::cleanup_dispatcher().registerDevice(
            infinicore::Device::Type::HYGON,
            &infinicore::adaptor::hygon_fa::mha_varlen::cleanup, true);
    });
}

#endif // ENABLE_HYGON_API && ENABLE_FLASH_ATTN_DLSYM
