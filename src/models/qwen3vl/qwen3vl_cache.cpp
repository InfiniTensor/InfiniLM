#include "qwen3vl_impl.hpp"

__C struct Qwen3vlCache *
createQwen3vlCache(const struct Qwen3vlModel *model) {
    Qwen3vlCache *cache = new Qwen3vlCache();
    auto ndev = model->dev_resources.size();
    auto nlayer = model->meta.text_meta.num_hidden_layers;
    auto max_len = model->meta.text_meta.max_tokens;
    auto dh = model->meta.text_meta.head_dim;
    auto nkv = model->meta.text_meta.num_key_value_heads / size_t(ndev);
    auto k_rot_shape = std::vector<size_t>{max_len, nkv, dh};
    auto v_shape = std::vector<size_t>{max_len, nkv, dh};
    for (size_t idev = 0; idev < ndev; idev++) {
        RUN_INFINI(infinirtSetDevice(model->device, model->dev_ids[idev]));
        auto k_rot_cache = std::vector<std::shared_ptr<Tensor>>();
        auto v_cache = std::vector<std::shared_ptr<Tensor>>();
        for (size_t layer = 0; layer < nlayer; layer++) {
            k_rot_cache.push_back(std::move(Tensor::buffer(model->meta.dtype, k_rot_shape)));
            v_cache.push_back(std::move(Tensor::buffer(model->meta.dtype, v_shape)));
        }
        cache->k_rot.push_back(k_rot_cache);
        cache->v.push_back(v_cache);
    }

    return cache;
}

//////还有visual deepstack需要cache？

__C void
dropQwen3vlCache(const struct Qwen3vlModel *model,
                    struct Qwen3vlCache *cache) {
    auto ndev = model->dev_resources.size();
    auto nlayer = model->meta.text_meta.num_hidden_layers;
    for (size_t idev = 0; idev < ndev; idev++) {
        RUN_INFINI(infinirtSetDevice(model->device, model->dev_ids[idev]));
        for (size_t layer = 0; layer < nlayer; layer++) {
            cache->k_rot[idev][layer].reset();
            cache->v[idev][layer].reset();
        }
    }
    delete cache;
}