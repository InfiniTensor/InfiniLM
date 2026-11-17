#if defined(ENABLE_ATEN) && defined(ENABLE_QY_API)
#include "infinicore/adaptor/denglin_fused_moe_adaptor.hpp"
#include "infinicore/adaptor/aten_adaptor.hpp"

#include <cstdlib>
#include <dlfcn.h>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace infinicore::adaptor::denglin_fused_moe {
namespace {

// Mangled name of free function in vllm._dl_C:
// invoke_fused_moe_opt(at::Tensor&, at::Tensor&, at::Tensor&,
//   optional const& x3, at::Tensor& x5, bool, long x4, bool x4,
//   vector<long> const&, long)
using invoke_fused_moe_opt_fn = void (*)(
    at::Tensor &,
    at::Tensor &,
    at::Tensor &,
    const std::optional<at::Tensor> &,
    const std::optional<at::Tensor> &,
    const std::optional<at::Tensor> &,
    at::Tensor &,
    at::Tensor &,
    at::Tensor &,
    at::Tensor &,
    at::Tensor &,
    bool,
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    bool,
    bool,
    bool,
    bool,
    const std::vector<int64_t> &,
    int64_t);

struct Symbols {
    void *handle = nullptr;
    invoke_fused_moe_opt_fn invoke = nullptr;
    std::string error;
};

Symbols &symbols() {
    static Symbols syms;
    static std::once_flag once;
    std::call_once(once, []() {
        const char *env_path = std::getenv("INFINICORE_DENGLIN_DL_C_SO");
        // Prefer exact override, then common dl-vLLM image layouts (3.12/3.10).
        const char *paths[] = {
            env_path,
            "/usr/local/lib/python3.12/dist-packages/vllm/_dl_C.cpython-312-x86_64-linux-gnu.so",
            "/usr/local/lib/python3.10/dist-packages/vllm/_dl_C.cpython-310-x86_64-linux-gnu.so",
            "/usr/lib/python3/dist-packages/vllm/_dl_C.cpython-312-x86_64-linux-gnu.so",
        };
        constexpr const char *kSym = "_Z20invoke_fused_moe_optRN2at6TensorES1_S1_RKSt8optionalIS0_ES5_S5_"
                                     "S1_S1_S1_S1_S1_bllllbbbbRKSt6vectorIlSaIlEEl";

        for (const char *path : paths) {
            if (path == nullptr || path[0] == '\0') {
                continue;
            }
            void *handle = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
            if (!handle) {
                syms.error = dlerror() ? dlerror() : "dlopen failed";
                continue;
            }
            auto fn = reinterpret_cast<invoke_fused_moe_opt_fn>(dlsym(handle, kSym));
            if (!fn) {
                syms.error = dlerror() ? dlerror() : "dlsym invoke_fused_moe_opt failed";
                dlclose(handle);
                continue;
            }
            syms.handle = handle;
            syms.invoke = fn;
            syms.error.clear();
            return;
        }
    });
    return syms;
}

} // namespace

bool available() {
    return symbols().invoke != nullptr;
}

void invoke_fused_moe_opt(
    at::Tensor &a,
    at::Tensor &b,
    at::Tensor &c,
    const std::optional<at::Tensor> &b_bias,
    const std::optional<at::Tensor> &b_scale,
    const std::optional<at::Tensor> &b_zp,
    at::Tensor &topk_weights,
    at::Tensor &topk_ids,
    at::Tensor &sorted_token_ids,
    at::Tensor &expert_ids,
    at::Tensor &num_tokens_post_padded,
    bool mul_routed_weight,
    int64_t top_k,
    int64_t block_size_m,
    int64_t block_size_n,
    int64_t block_size_k,
    bool use_fp8_w8a8,
    bool use_int8_w8a16,
    bool use_int4_w4a16,
    bool use_mxfp4_w4a16,
    const std::vector<int64_t> &block_shape,
    int64_t m) {
    auto &syms = symbols();
    if (!syms.invoke) {
        throw std::runtime_error(
            "Denglin fused MoE unavailable (vllm._dl_C.invoke_fused_moe_opt): " + syms.error);
    }
    infinicore::adaptor::set_aten_stream_to_infinicore();
    syms.invoke(
        a, b, c, b_bias, b_scale, b_zp,
        topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded,
        mul_routed_weight, top_k, block_size_m, block_size_n, block_size_k,
        use_fp8_w8a8, use_int8_w8a16, use_int4_w4a16, use_mxfp4_w4a16,
        block_shape, m);
}

} // namespace infinicore::adaptor::denglin_fused_moe
#endif
