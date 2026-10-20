#pragma once
#include "cache/base_cache.hpp"
#include <stdexcept>
namespace infinilm::cache {
class NativeSparseCacheConfig final:public CacheConfig {
public:
 NativeSparseCacheConfig(size_t rows,size_t capacity):rows_(rows),capacity_(capacity){if(!rows||rows>128||!capacity||capacity>4096)throw std::invalid_argument("Native sparse cache requires rows1..128 and capacity1..4096");}
 size_t rows()const{return rows_;}size_t capacity()const{return capacity_;}
 std::unique_ptr<CacheConfig>unique_copy()const override{return std::make_unique<NativeSparseCacheConfig>(rows_,capacity_);}
private:size_t rows_,capacity_;
};
}
