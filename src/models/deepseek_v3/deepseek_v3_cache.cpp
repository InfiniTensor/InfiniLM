#include "deepseek_v3_impl.hpp"

__C struct DeepSeekV3Cache *
createDeepSeekV3Cache(const struct DeepSeekV3Model *model) {
    DeepSeekV3Cache *cache = new DeepSeekV3Cache();
    auto ndev = model->dev_resources.size();
    auto nlayer = model->meta.n_dense_layer + model->meta.n_sparse_layer;
    auto max_len = model->meta.dctx;
    auto d_rope = model->meta.d_rope;
    auto d_nope = model->meta.d_nope;
    auto d_v = model->meta.d_v;
    auto nh = model->meta.nh;
    auto kshape = std::vector<size_t>{max_len, d_rope};
    auto cshape = std::vector<size_t>{max_len, nh * (d_nope + d_v)};
    for (size_t idev = 0; idev < ndev; idev++) {
        RUN_INFINI(infinirtSetDevice(model->device, model->dev_ids[idev]));
        auto kcache = std::vector<std::shared_ptr<Tensor>>();
        auto ccache = std::vector<std::shared_ptr<Tensor>>();
        for (size_t layer = 0; layer < nlayer; layer++) {
            kcache.push_back(std::move(Tensor::buffer(model->meta.dt_logits, kshape)));
            ccache.push_back(std::move(Tensor::buffer(model->meta.dt_logits, cshape)));
        }
        cache->k.push_back(kcache);
        cache->c.push_back(ccache);
    }

    return cache;
}

__C void
dropDeepSeekV3Cache(const struct DeepSeekV3Model *model,
                    struct DeepSeekV3Cache *cache) {
    auto ndev = model->dev_resources.size();
    auto nlayer = model->meta.n_dense_layer + model->meta.n_sparse_layer;
    for (size_t idev = 0; idev < ndev; idev++) {
        RUN_INFINI(infinirtSetDevice(model->device, model->dev_ids[idev]));
        for (size_t layer = 0; layer < nlayer; layer++) {
            cache->k[idev][layer].reset();
            cache->c[idev][layer].reset();
        }
    }
    delete cache;
}