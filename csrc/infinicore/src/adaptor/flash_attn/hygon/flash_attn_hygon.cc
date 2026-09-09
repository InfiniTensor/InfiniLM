#if defined(ENABLE_HYGON_API) && defined(ENABLE_FLASH_ATTN)

#include "flash_attn_hygon.hpp"

#include <ATen/ATen.h>
#include <c10/util/Optional.h>
#include <dlfcn.h>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using mha_fwd_kvcache_fn_t = std::vector<at::Tensor> (*)(
    at::Tensor &q,
    const at::Tensor &kcache,
    const at::Tensor &vcache,
    std::optional<const at::Tensor> &k_,
    std::optional<const at::Tensor> &v_,
    std::optional<const at::Tensor> &seqlens_k_,
    std::optional<const at::Tensor> &rotary_cos_,
    std::optional<const at::Tensor> &rotary_sin_,
    std::optional<const at::Tensor> &cache_batch_idx_,
    std::optional<const at::Tensor> &leftpad_k_,
    std::optional<at::Tensor> &block_table_,
    std::optional<at::Tensor> &alibi_slopes_,
    std::optional<at::Tensor> &out_,
    float softmax_scale,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    float softcap,
    bool is_rotary_interleaved,
    int num_splits,
    const std::optional<at::Tensor> &s_aux_);

using mha_varlen_fwd_fn_t = std::vector<at::Tensor> (*)(
    at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    std::optional<at::Tensor> &out_,
    const at::Tensor &cu_seqlens_q,
    const at::Tensor &cu_seqlens_k,
    std::optional<at::Tensor> &seqused_k,
    std::optional<const at::Tensor> &leftpad_k_,
    std::optional<at::Tensor> &block_table_,
    std::optional<at::Tensor> &alibi_slopes_,
    int max_seqlen_q,
    int max_seqlen_k,
    float p_dropout,
    float softmax_scale,
    bool zero_tensors,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    float softcap,
    bool return_softmax,
    std::optional<at::Tensor> q_descale_,
    std::optional<at::Tensor> k_descale_,
    std::optional<at::Tensor> v_descale_,
    std::optional<at::Generator> gen_,
    const std::optional<at::Tensor> &s_aux_);

namespace {

void *resolve_flash_symbol(const char *name) {
    void *sym = dlsym(RTLD_DEFAULT, name);
    if (!sym) {
        throw std::runtime_error(std::string("flash_attn symbol not found: ") + name);
    }
    return sym;
}

void *resolve_flash_extension_symbol(const char *name) {
    static std::string last_error;
    static void *handle = [&]() -> void * {
        const char *candidates[] = {
            "flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so",
            "/usr/local/lib/python3.10/dist-packages/flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so",
        };
        for (const char *candidate : candidates) {
            dlerror();
            if (void *h = dlopen(candidate, RTLD_NOW | RTLD_GLOBAL)) {
                return h;
            }
            if (const char *err = dlerror()) {
                last_error += std::string(candidate) + ": " + err + "; ";
            }
        }
        return nullptr;
    }();
    if (handle) {
        dlerror();
        if (void *sym = dlsym(handle, name)) {
            return sym;
        }
        if (const char *err = dlerror()) {
            throw std::runtime_error(std::string("flash_attn extension symbol not found: ") + name + "; " + err);
        }
        throw std::runtime_error(std::string("flash_attn extension symbol not found: ") + name);
    }
    throw std::runtime_error(std::string("flash_attn extension dlopen failed for symbol: ") + name + "; " + last_error);
}

} // namespace

namespace flash {

std::vector<at::Tensor>
mha_fwd_kvcache(at::Tensor &q,
                const at::Tensor &kcache,
                const at::Tensor &vcache,
                std::optional<const at::Tensor> &k_,
                std::optional<const at::Tensor> &v_,
                std::optional<const at::Tensor> &seqlens_k_,
                std::optional<const at::Tensor> &rotary_cos_,
                std::optional<const at::Tensor> &rotary_sin_,
                std::optional<const at::Tensor> &cache_batch_idx_,
                std::optional<const at::Tensor> &leftpad_k_,
                std::optional<at::Tensor> &block_table_,
                std::optional<at::Tensor> &alibi_slopes_,
                std::optional<at::Tensor> &out_,
                float softmax_scale,
                bool is_causal,
                int window_size_left,
                int window_size_right,
                float softcap,
                bool is_rotary_interleaved,
                int num_splits) {
    static auto fn = reinterpret_cast<mha_fwd_kvcache_fn_t>(resolve_flash_extension_symbol("mha_fwd_kvcache"));
    std::optional<at::Tensor> s_aux = std::nullopt;
    return fn(q, kcache, vcache, k_, v_, seqlens_k_, rotary_cos_, rotary_sin_,
              cache_batch_idx_, leftpad_k_, block_table_, alibi_slopes_, out_,
              softmax_scale, is_causal, window_size_left, window_size_right,
              softcap, is_rotary_interleaved, num_splits, s_aux);
}

std::vector<at::Tensor>
vllm_mha_varlen_fwd(at::Tensor &q,
                    const at::Tensor &k,
                    const at::Tensor &v,
                    std::optional<at::Tensor> &out_,
                    const at::Tensor &cu_seqlens_q,
                    const at::Tensor &cu_seqlens_k,
                    std::optional<at::Tensor> &seqused_k,
                    std::optional<const at::Tensor> &leftpad_k_,
                    std::optional<at::Tensor> &block_table_,
                    std::optional<at::Tensor> &alibi_slopes_,
                    int max_seqlen_q,
                    int max_seqlen_k,
                    float p_dropout,
                    float softmax_scale,
                    bool zero_tensors,
                    bool is_causal,
                    int window_size_left,
                    int window_size_right,
                    float softcap,
                    bool return_softmax,
                    std::optional<at::Generator> gen_) {
    static auto fn = reinterpret_cast<mha_varlen_fwd_fn_t>(resolve_flash_extension_symbol("mha_varlen_fwd"));
    std::optional<at::Tensor> q_descale = std::nullopt;
    std::optional<at::Tensor> k_descale = std::nullopt;
    std::optional<at::Tensor> v_descale = std::nullopt;
    std::optional<at::Tensor> s_aux = std::nullopt;
    return fn(q, k, v, out_, cu_seqlens_q, cu_seqlens_k, seqused_k, leftpad_k_,
              block_table_, alibi_slopes_, max_seqlen_q, max_seqlen_k, p_dropout,
              softmax_scale, zero_tensors, is_causal, window_size_left,
              window_size_right, softcap, return_softmax, q_descale, k_descale,
              v_descale, gen_, s_aux);
}

} // namespace flash

#endif
