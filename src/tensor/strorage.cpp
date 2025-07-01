#include "../allocator.hpp"
#include "../tensor.hpp"

std::shared_ptr<Storage> Storage::create(size_t size) {
    auto storage = std::shared_ptr<Storage>(new Storage());
    RUN_INFINI(infinirtMalloc(&storage->_memory, size));
    storage->_size = size;
    RUN_INFINI(infinirtGetDevice(&storage->_device_type, &storage->_device_id));
    return storage;
}

std::shared_ptr<Storage> Storage::createAsync(size_t size, infinirtStream_t stream) {
    auto storage = std::shared_ptr<Storage>(new Storage());
    RUN_INFINI(infinirtMallocAsync(&storage->_memory, size, stream));
    storage->_size = size;
    RUN_INFINI(infinirtGetDevice(&storage->_device_type, &storage->_device_id));
    return storage;
}

std::shared_ptr<Storage> Storage::createFromPool(size_t size, std::shared_ptr<MemoryPool> pool) {
    auto storage = std::shared_ptr<Storage>(new Storage());
    storage->_memory_pool = pool;
    if (pool) {
        storage->_memory = pool->alloc(size);
    } else {
        RUN_INFINI(infinirtMalloc(&storage->_memory, size));
    }
    storage->_size = size;
    RUN_INFINI(infinirtGetDevice(&storage->_device_type, &storage->_device_id));
    return storage;
}

std::shared_ptr<Storage> Storage::createHost(size_t size) {
    auto storage = std::shared_ptr<Storage>(new Storage());
    RUN_INFINI(infinirtMallocHost(&storage->_memory, size));
    storage->_size = size;
    storage->_device_type = INFINI_DEVICE_CPU;
    storage->_device_id = 0;
    storage->_memory_pool = nullptr; // No pool for host memory
    return storage;
}

Storage::~Storage() {
    if (_memory_pool) {
        _memory_pool->release(_memory); 
    } else {
        if (_device_type == INFINI_DEVICE_CPU) {
            RUN_INFINI(infinirtFreeHost(_memory));
        } else {
            RUN_INFINI(infinirtFree(_memory));
        }
    }
}
