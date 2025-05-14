#include "../tensor.hpp"

std::shared_ptr<Storage> Storage::create(size_t size) {
    auto storage = std::make_shared<Storage>();
    RUN_INFINI(infinirtMalloc(&storage->memory, size));
    storage->size = size;
    RUN_INFINI(infinirtGetDevice(&storage->device_type, &storage->device_id));
    return storage;
}

std::shared_ptr<Storage> Storage::createAsync(size_t size, infinirtStream_t stream) {
    auto storage = std::make_shared<Storage>();
    RUN_INFINI(infinirtMallocAsync(&storage->memory, size, stream));
    storage->size = size;
    RUN_INFINI(infinirtGetDevice(&storage->device_type, &storage->device_id));
    return storage;
}

std::shared_ptr<Storage> Storage::createHost(size_t size) {
    auto storage = std::make_shared<Storage>();
    RUN_INFINI(infinirtMallocHost(&storage->memory, size));
    storage->size = size;
    storage->device_type = INFINI_DEVICE_CPU;
    storage->device_id = 0;
    return storage;
}

Storage::~Storage() {
    if (device_type == INFINI_DEVICE_CPU) {
        RUN_INFINI(infinirtFreeHost(memory));
    } else {
        RUN_INFINI(infinirtFree(memory));
    }
}
