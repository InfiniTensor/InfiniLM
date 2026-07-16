#pragma once

#include <cstddef>
#include <functional>
#include <iostream>
#include <list>
#include <optional>
#include <stdexcept>
#include <unordered_map>

namespace infinicore::common {
template <typename Key, typename Value>
class LRUCache {
public:
    using KeyValuePair = std::pair<Key, Value>;
    using ListIt = typename std::list<KeyValuePair>::iterator;
    using Destructor = std::function<void(Value &)>;

    explicit LRUCache(size_t capacity = 100, Destructor destructor = nullptr)
        : capacity_(capacity), destructor_(destructor) {
        if (capacity == 0) {
            capacity_ = UINT64_MAX; // effectively unbounded
        }
    }

    ~LRUCache() {
        cleanup();
    }

    bool contains(const Key &key) const {
        return map_.find(key) != map_.end();
    }

    void put(const Key &key, const Value &value) {
        auto it = map_.find(key);
        if (it != map_.end()) {
            if (destructor_) {
                destructor_(it->second->second);
            }
            it->second->second = value;
            touch(it);
        } else {
            // insert new
            if (list_.size() >= capacity_) {
                evictLRU();
            }
            list_.emplace_front(key, value);
            map_[key] = list_.begin();
        }
    }

    std::optional<Value> get(const Key &key) {
        auto it = map_.find(key);
        if (it == map_.end()) {
            return std::nullopt;
        }
        touch(it);
        return it->second->second;
    }

    std::optional<Value> get(const Key &key) const {
        auto it = map_.find(key);
        if (it == map_.end()) {
            return std::nullopt;
        }
        // Note: can't touch in const context
        return it->second->second;
    }

    void setDestructor(Destructor destructor) {
        destructor_ = destructor;
    }

    void setCapacity(size_t capacity) {
        capacity_ = capacity;
        while (list_.size() > capacity_) {
            evictLRU();
        }
    }

    void clear() {
        if (destructor_) {
            for (auto &item : list_) {
                safeDestruct(item.second);
            }
        }
        list_.clear();
        map_.clear();
    }

    const std::list<KeyValuePair> &getAllItems() const {
        return list_;
    }

protected:
    std::list<KeyValuePair> list_; // front = most recent, back = least

private:
    void touch(typename std::unordered_map<Key, ListIt>::iterator it) {
        // move this key to front (most recent)
        list_.splice(list_.begin(), list_, it->second);
        it->second = list_.begin();
    }

    void safeDestruct(Value &value) {
        if (!destructor_) {
            return;
        }

        try {
            destructor_(value);
        } catch (const std::exception &e) {
            // Built-in default error handling
            std::cerr << "Cache destructor error (type: " << typeid(Value).name()
                      << "): " << e.what() << std::endl;
        }
    }

    void evictLRU() {
        if (!list_.empty()) {
            auto &kv = list_.back();
            safeDestruct(kv.second);
            map_.erase(kv.first);
            list_.pop_back();
        }
    }

    void cleanup() {
        clear();
    }

    size_t capacity_;
    std::unordered_map<Key, ListIt> map_;
    Destructor destructor_;
};

} // namespace infinicore::common
