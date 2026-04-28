#pragma once

#include "../utils/infini_status_string.h"

#include <spdlog/cfg/env.h>
#include <spdlog/spdlog.h>
#include <stdexcept>

inline struct SpdlogInitializer {
    SpdlogInitializer() {
        if (!std::getenv("INFINICORE_LOG_LEVEL")) {
            spdlog::set_level(spdlog::level::info);
        } else {
            spdlog::cfg::load_env_levels("INFINICORE_LOG_LEVEL");
        }
        // Set pattern for logging
        // Using SPDLOG_* macros enables source location support (%s and %#)
        // Format: [timestamp] [level] [file:line] message
        spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%s:%#] %v");
    }
} spdlog_initializer;

#define STRINGIZE_(x) #x
#define STRINGIZE(x) STRINGIZE_(x)

#define INFINICORE_CHECK_ERROR(call)                                                                            \
    do {                                                                                                        \
        SPDLOG_DEBUG("Entering `" #call "` at `" __FILE__ ":" STRINGIZE(__LINE__) "`.");                        \
        infiniStatus_t ret = (call);                                                                            \
        SPDLOG_DEBUG("Exiting `" #call "` at `" __FILE__ ":" STRINGIZE(__LINE__) "`.");                         \
        if (ret != INFINI_STATUS_SUCCESS) {                                                                     \
            throw std::runtime_error("`" #call "` failed with error: " + std::string(infini_status_string(ret)) \
                                     + " from " + std::string(__func__)                                         \
                                     + " at " + std::string(__FILE__)                                           \
                                     + ":" + std::to_string(__LINE__) + ".");                                   \
        }                                                                                                       \
    } while (false)

#define INFINICORE_ASSERT_TENSORS_SAME_DEVICE(FIRST___, ...)                      \
    do {                                                                          \
        const auto &first_device___ = (FIRST___)->device();                       \
        for (const auto &tensor___ : {__VA_ARGS__}) {                             \
            if (first_device___ != (tensor___)->device()) {                       \
                throw std::runtime_error("Tensor devices mismatch "               \
                                         + first_device___.toString() + " vs "    \
                                         + (tensor___)->device().toString()       \
                                         + " from " + std::string(__func__)       \
                                         + " at " + std::string(__FILE__)         \
                                         + ":" + std::to_string(__LINE__) + "."); \
            }                                                                     \
        }                                                                         \
    } while (0)

#define INFINICORE_ASSERT(CONDITION__)                                                                                                         \
    do {                                                                                                                                       \
        if (!(CONDITION__)) {                                                                                                                  \
            SPDLOG_ERROR(                                                                                                                      \
                "Assertion `{}` failed from {} at {}:{}",                                                                                      \
                #CONDITION__, __func__, __FILE__, __LINE__);                                                                                   \
            throw std::runtime_error(                                                                                                          \
                std::string("Assertion `") + #CONDITION__ + "` failed from " + __func__ + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
        }                                                                                                                                      \
    } while (0)
