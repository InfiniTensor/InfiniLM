#pragma once

#include "../backends/attention_backends.hpp"
#include "../cache/cache.hpp"
#include "model_config.hpp"
#include <memory>

namespace infinilm::config {
/*
https://github.com/vllm-project/vllm/blob/main/vllm/config/vllm.py

@config(config=ConfigDict(arbitrary_types_allowed=True))
class VllmConfig:
    """Dataclass which contains all vllm-related configuration. This
    simplifies passing around the distinct configurations in the codebase.
    """

    # TODO: use default_factory once default constructing ModelConfig doesn't try to download a model
    model_config: ModelConfig = Field(default=None)

    """Model configuration."""
    cache_config: CacheConfig = Field(default_factory=CacheConfig)

    """Cache configuration."""
    parallel_config: ParallelConfig = Field(default_factory=ParallelConfig)

    pass
*/
struct InfinilmConfig {
public:
    InfinilmConfig() = default;
    InfinilmConfig(const infinilm::backends::AttentionBackend &backend,
                   const std::shared_ptr<infinilm::config::ModelConfig> &model_config,
                   const cache::CacheConfig *cache_config)
        : attention_backend(backend),
          model_config(model_config),
          cache_config(cache_config) {}

public:
    infinilm::backends::AttentionBackend attention_backend;
    std::shared_ptr<infinilm::config::ModelConfig> model_config;
    const cache::CacheConfig *cache_config;
};

/*
@contextmanager
def set_current_vllm_config(
    vllm_config: VllmConfig, check_compile=False, prefix: str | None = None
):
    """
    Temporarily set the current vLLM config.
    Used during model initialization.
    We save the current vLLM config in a global variable,
    so that all modules can access it, e.g. custom ops
    can access the vLLM config to determine how to dispatch.
    """
    global _current_vllm_config, _current_prefix
    old_vllm_config = _current_vllm_config

*/
void set_current_infinilm_config(const InfinilmConfig &config);

/*
def get_current_vllm_config() -> VllmConfig:
    if _current_vllm_config is None:
        raise AssertionError(
            "Current vLLM config is not set. This typically means "
            "get_current_vllm_config() was called outside of a "
            "set_current_vllm_config() context, or a CustomOp was instantiated "
            "at module import time or model forward time when config is not set. "
            "For tests that directly test custom ops/modules, use the "
            "'default_vllm_config' pytest fixture from tests/conftest.py."
        )
    return _current_vllm_config

*/
const InfinilmConfig &get_current_infinilm_config();

} // namespace infinilm::config
