from infinicore.lib import _infinicore


def kv_caching(k_cache, v_cache, k, v, past_kv_lengths):
    _infinicore.kv_caching_(
        k_cache._underlying,
        v_cache._underlying,
        k._underlying,
        v._underlying,
        past_kv_lengths._underlying,
    )

    return k_cache, v_cache
