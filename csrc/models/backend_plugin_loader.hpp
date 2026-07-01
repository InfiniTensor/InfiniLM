#pragma once

#include <string>
#include <vector>

namespace infinilm::models {

/**
 * Load one out-of-tree backend plugin shared object.
 *
 * The plugin may either rely on static initializers that call
 * `register_causal_lm_model` / `register_model_config`, or export an optional
 * `extern "C" void infinilm_backend_plugin_init()` function. Loading is
 * idempotent for each path.
 */
void load_backend_plugin(const std::string &plugin_path);

/**
 * Load backend plugins from `INFINILM_BACKEND_PLUGINS`.
 *
 * The environment variable accepts comma-separated shared object paths.
 */
void load_backend_plugins_from_env();

/**
 * Return plugin paths that have already been loaded.
 */
std::vector<std::string> loaded_backend_plugins();

} // namespace infinilm::models
