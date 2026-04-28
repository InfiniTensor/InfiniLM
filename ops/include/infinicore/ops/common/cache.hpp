#pragma once

#include "../../common/LRUCache.hpp"
#include "../../context/context.hpp"
#include <array>
#include <functional>
#include <memory>
#include <vector>

namespace infinicore::op::common {

template <typename Key, typename Value>
class OpCache {
private:
    using BaseCache = infinicore::common::LRUCache<Key, Value>;
    using Destructor = typename BaseCache::Destructor;
    using CacheVector = std::vector<BaseCache>;

public:
    explicit OpCache(size_t capacity = 100, Destructor destructor = nullptr)
        : capacity_(capacity), destructor_(destructor) {}

    ~OpCache() {
        clear();
    }

    BaseCache &getCache(Device::Type device_type, size_t device_index) {
        auto &cache_vector = caches_[static_cast<size_t>(device_type)];

        if (cache_vector.size() <= device_index) {
            cache_vector.resize(device_index + 1, BaseCache(capacity_, destructor_));
        } else {
            cache_vector[device_index].setDestructor(destructor_);
        }

        return cache_vector[device_index];
    }

    BaseCache &getCache(Device device) {
        return getCache(device.getType(), device.getIndex());
    }

    void setCapacity(size_t capacity) {
        capacity_ = capacity;
        for (auto &vec : caches_) {
            for (auto &cache : vec) {
                cache.setCapacity(capacity);
            }
        }
    }

    void clear() {
        Device current_device = context::getDevice();

        for (size_t type_idx = 0; type_idx < caches_.size(); ++type_idx) {
            auto &vec = caches_[type_idx];
            for (size_t dev_idx = 0; dev_idx < vec.size(); ++dev_idx) {
                Device target_device(static_cast<Device::Type>(type_idx), dev_idx);

                if (current_device != target_device) {
                    context::setDevice(target_device);
                }

                vec[dev_idx].clear();

                if (current_device != target_device) {
                    context::setDevice(current_device);
                }
            }
            vec.clear();
        }

        caches_ = {};
    }

private:
    size_t capacity_;
    Destructor destructor_;

    std::array<CacheVector, static_cast<size_t>(Device::Type::COUNT)> caches_ = {};
};

} // namespace infinicore::op::common
