#include "../../weights/weights.hpp"
#include "attention.hpp"

#include <stdexcept>
#include <exception>
#include "runtime/runtime.hpp"

namespace infinilm::layers::attention {

void init_kv_cache_quant_params(infinilm::runtime::Runtime &runtime,
                                std::function<void(const std::string &, infinilm::weights::Weight)> register_fn,
                                infinilm::weights::Weight &kv_cache_k_scale,
                                infinilm::weights::Weight &kv_cache_v_scale,
                                infinilm::quantization::KVQuantAlgo kv_quant_scheme) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("KV scale construction requires eager Runtime");
    switch (kv_quant_scheme) {
    case infinilm::quantization::KVQuantAlgo::NONE:
        return;
    case infinilm::quantization::KVQuantAlgo::INT8:
        try {
            kv_cache_k_scale = infinilm::weights::Weight(runtime, {1}, infinilm::runtime::DataType::kFloat32, 0, 0, 1);
            register_fn("kv_cache_k_scale", kv_cache_k_scale);
            kv_cache_v_scale = infinilm::weights::Weight(runtime, {1}, infinilm::runtime::DataType::kFloat32, 0, 0, 1);
            register_fn("kv_cache_v_scale", kv_cache_v_scale);
        } catch (...) {
            // References are caller owning members, not helper-local backing.
            try { runtime.syncStream(); } catch (...) { std::terminate(); }
            throw;
        }
        return;
    default:
        throw std::runtime_error("unsupported kv_quant_scheme");
    }
}

} // namespace infinilm::layers::attention
