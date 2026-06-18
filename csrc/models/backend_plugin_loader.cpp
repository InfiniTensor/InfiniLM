#include "backend_plugin_loader.hpp"

#include <cstdlib>
#include <dlfcn.h>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace infinilm::models {
namespace {

using PluginInitFn = void (*)();

std::mutex &loader_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::unordered_map<std::string, void *> &loaded_handles() {
    static std::unordered_map<std::string, void *> handles;
    return handles;
}

std::string trim(std::string value) {
    const auto begin = value.find_first_not_of(" \t\n\r");
    if (begin == std::string::npos) {
        return "";
    }
    const auto end = value.find_last_not_of(" \t\n\r");
    return value.substr(begin, end - begin + 1);
}

std::vector<std::string> split_plugins(const char *env_value) {
    std::vector<std::string> plugins;
    if (env_value == nullptr || *env_value == '\0') {
        return plugins;
    }

    std::stringstream stream(env_value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        item = trim(item);
        if (!item.empty()) {
            plugins.push_back(item);
        }
    }
    return plugins;
}

} // namespace

void load_backend_plugin(const std::string &plugin_path) {
    const std::string path = trim(plugin_path);
    if (path.empty()) {
        return;
    }

    std::lock_guard<std::mutex> lock(loader_mutex());
    auto &handles = loaded_handles();
    if (handles.find(path) != handles.end()) {
        return;
    }

    void *handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (handle == nullptr) {
        const char *error = dlerror();
        throw std::runtime_error(
            "infinilm::models::load_backend_plugin: failed to load " + path
            + ": " + (error == nullptr ? "unknown dlopen error" : std::string(error)));
    }

    dlerror();
    auto init_fn = reinterpret_cast<PluginInitFn>(dlsym(handle, "infinilm_backend_plugin_init"));
    const char *dlsym_error = dlerror();
    if (dlsym_error == nullptr && init_fn != nullptr) {
        init_fn();
    }

    handles[path] = handle;
}

void load_backend_plugins_from_env() {
    for (const auto &plugin : split_plugins(std::getenv("INFINILM_BACKEND_PLUGINS"))) {
        load_backend_plugin(plugin);
    }
}

std::vector<std::string> loaded_backend_plugins() {
    std::lock_guard<std::mutex> lock(loader_mutex());
    std::vector<std::string> plugins;
    plugins.reserve(loaded_handles().size());
    for (const auto &[path, _] : loaded_handles()) {
        plugins.push_back(path);
    }
    return plugins;
}

} // namespace infinilm::models
