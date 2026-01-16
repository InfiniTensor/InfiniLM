#ifndef KV_COMPRESSION_H
#define KV_COMPRESSION_H

#include <stdint.h>

#include <infinirt.h>

struct KVCache;

typedef struct {
    uint32_t enable;
    uint32_t compression_factor;
    uint32_t min_seq_len;
    uint32_t image_kv_len;
    const char *weight_path; // path to .bin weights (see docs/KVCacheCompressionWeightFormat.md)
} KVCompressionConfig;

// Compress KVCache in-place:
// - Reads KV from [0, seq_len) and writes compressed KV back into the same cache prefix [0, new_len).
// - Returns new_len on success; returns seq_len on no-op/failure.
__C __export uint32_t
compressKVCacheInplace(struct KVCache *kv_cache, uint32_t seq_len, const KVCompressionConfig *cfg);

#endif // KV_COMPRESSION_H

