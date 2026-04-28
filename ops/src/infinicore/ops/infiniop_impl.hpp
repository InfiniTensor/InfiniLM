#pragma once

#include "../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include <infiniop.h>

#define INFINIOP_CACHABLE_DESCRIPTOR(__DESC_TYPE__, __OP_NAME__, __SIZE__) \
    struct __DESC_TYPE__ {                                                 \
        infiniop##__OP_NAME__##Descriptor_t desc = nullptr;                \
                                                                           \
        explicit __DESC_TYPE__(infiniop##__OP_NAME__##Descriptor_t d)      \
            : desc(d) {}                                                   \
                                                                           \
        /* non-copyable */                                                 \
        __DESC_TYPE__(const __DESC_TYPE__ &) = delete;                     \
        __DESC_TYPE__ &operator=(const __DESC_TYPE__ &) = delete;          \
                                                                           \
        /* movable */                                                      \
        __DESC_TYPE__(__DESC_TYPE__ &&other) noexcept                      \
            : desc(other.desc) {                                           \
            other.desc = nullptr;                                          \
        }                                                                  \
                                                                           \
        __DESC_TYPE__ &operator=(__DESC_TYPE__ &&other) noexcept {         \
            if (this != &other) {                                          \
                if (desc != nullptr) {                                     \
                    infiniopDestroy##__OP_NAME__##Descriptor(desc);        \
                }                                                          \
                desc = other.desc;                                         \
                other.desc = nullptr;                                      \
            }                                                              \
            return *this;                                                  \
        }                                                                  \
                                                                           \
        ~__DESC_TYPE__() {                                                 \
            if (desc != nullptr) {                                         \
                infiniopDestroy##__OP_NAME__##Descriptor(desc);            \
            }                                                              \
        }                                                                  \
    };                                                                     \
                                                                           \
    thread_local common::OpCache<size_t, std::shared_ptr<__DESC_TYPE__>>   \
        caches(                                                            \
            __SIZE__,                                                      \
            [](std::shared_ptr<__DESC_TYPE__> &desc) {                     \
                desc = nullptr;                                            \
            });

#define INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(__DESC_TYPE__, __DESC_NAME__, __INFINIOP_NAME__, __HASH_KEY__, ...) \
    std::shared_ptr<__DESC_TYPE__> __DESC_NAME__;                                                                      \
    {                                                                                                                  \
        auto device__ = context::getDevice();                                                                          \
        auto &cache__ = caches.getCache(device__);                                                                     \
        __DESC_NAME__ = cache__.get(__HASH_KEY__).value_or(nullptr);                                                   \
        if (!__DESC_NAME__) {                                                                                          \
            __DESC_NAME__ = std::make_shared<__DESC_TYPE__>(nullptr);                                                  \
            INFINICORE_CHECK_ERROR(infiniopCreate##__INFINIOP_NAME__##Descriptor(                                      \
                context::getInfiniopHandle(device__),                                                                  \
                &__DESC_NAME__->desc,                                                                                  \
                __VA_ARGS__));                                                                                         \
            cache__.put(__HASH_KEY__, __DESC_NAME__);                                                                  \
        }                                                                                                              \
    }

#define INFINIOP_WORKSPACE_TENSOR(__TENSOR_NAME__, __INFINIOP_NAME__, __DESC_NAME__)                                 \
    Tensor __TENSOR_NAME__;                                                                                          \
    {                                                                                                                \
        auto device__ = context::getDevice();                                                                        \
        size_t workspace_size = 0;                                                                                   \
        INFINICORE_CHECK_ERROR(infiniopGet##__INFINIOP_NAME__##WorkspaceSize(__DESC_NAME__->desc, &workspace_size)); \
        __TENSOR_NAME__ = Tensor::empty({workspace_size}, DataType::U8, device__);                                   \
    }
