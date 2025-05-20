#include "jiuge_impl.hpp"

__C struct KVCache *createKVCache(const JiugeModel *model) {
    KVCache *cache = new KVCache();
    auto ndev = model->dev_resources.size();
    auto nkvh = model->meta.nkvh / ndev;
    auto max_len = model->meta.dctx;
    auto dh = model->meta.dh;
    auto shape = std::vector<size_t>{nkvh, max_len, dh};
    for (unsigned int idev = 0; idev < ndev; idev++) {
        RUN_INFINI(infinirtSetDevice(model->device, model->dev_ids[idev]));
        auto kcache = std::vector<std::shared_ptr<Tensor>>();
        auto vcache = std::vector<std::shared_ptr<Tensor>>();
        for (unsigned int layer = 0; layer < model->meta.nlayer; layer++) {
            kcache.push_back(std::move(Tensor::buffer(model->meta.dt_logits, shape)));
            vcache.push_back(std::move(Tensor::buffer(model->meta.dt_logits, shape)));
        }
        cache->k.push_back(kcache);
        cache->v.push_back(vcache);
    }

    return cache;
}

__C struct KVCache *duplicateKVCache(const JiugeModel *model,
                                     const KVCache *kv_cache,
                                     unsigned int seq_len) {
    auto new_kv_cache = createKVCache(model);
    auto ndev = model->dev_resources.size();
    for (unsigned int idev = 0; idev < ndev; idev++) {
        RUN_INFINI(infinirtSetDevice(model->device, model->dev_ids[idev]));
        for (unsigned int layer = 0; layer < model->meta.nlayer; layer++) {
            new_kv_cache->k[idev][layer]
                ->slice(1, 0, seq_len)
                ->copyFrom(kv_cache->k[idev][layer]->slice(1, 0, seq_len),
                           model->dev_resources[idev].handle);

            new_kv_cache->v[idev][layer]
                ->slice(1, 0, seq_len)
                ->copyFrom(kv_cache->v[idev][layer]->slice(1, 0, seq_len),
                           model->dev_resources[idev].handle);
        }
    }
    return new_kv_cache;
}

__C void dropKVCache(JiugeModel const *model, KVCache *kv_cache) {
    auto ndev = model->dev_resources.size();
    for (unsigned int idev = 0; idev < ndev; idev++) {
        RUN_INFINI(infinirtSetDevice(model->device, model->dev_ids[idev]));
        for (unsigned int layer = 0; layer < model->meta.nlayer; layer++) {
            kv_cache->k[idev][layer].reset();
            kv_cache->v[idev][layer].reset();
        }
    }
    delete kv_cache;
}
