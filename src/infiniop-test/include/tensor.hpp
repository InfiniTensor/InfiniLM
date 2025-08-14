#ifndef __INFINIOPTEST_TENSOR_HPP__
#define __INFINIOPTEST_TENSOR_HPP__
#include "file_mapping.hpp"
#include "gguf.hpp"
#include <infiniop.h>

inline infiniDtype_t ggmlTypeToInfiniType(GGML_TYPE type) {
    switch (type) {
    case GGML_TYPE_Q8_K:
        return INFINI_DTYPE_BOOL;
    case GGML_TYPE_I8:
        return INFINI_DTYPE_I8;
    case GGML_TYPE_I16:
        return INFINI_DTYPE_I16;
    case GGML_TYPE_I32:
        return INFINI_DTYPE_I32;
    case GGML_TYPE_I64:
        return INFINI_DTYPE_I64;
    case GGML_TYPE_BF16:
        return INFINI_DTYPE_BF16;
    case GGML_TYPE_F16:
        return INFINI_DTYPE_F16;
    case GGML_TYPE_F32:
        return INFINI_DTYPE_F32;
    case GGML_TYPE_F64:
        return INFINI_DTYPE_F64;
    default:
        throw std::runtime_error("Unsupported GGML type");
    }
}

namespace infiniop_test {
class Memory {
private:
    void *_ptr;
    size_t _size;
    infiniDevice_t _device;
    int _device_id;
    std::shared_ptr<FileMapping> _file_mapping;

public:
    Memory(size_t size, infiniDevice_t device, int device_id);
    Memory(const std::shared_ptr<FileMapping> &file_mapping, void *ptr, size_t size);
    ~Memory();
    void *ptr() const { return _ptr; }
    size_t size() const { return _size; }
    infiniDevice_t device() const { return _device; }
    int device_id() const { return _device_id; }
};

class Tensor {
private:
    infiniopTensorDescriptor_t _desc;
    std::shared_ptr<Memory> _memory;
    std::vector<size_t> _shape;
    std::vector<ptrdiff_t> _strides;
    size_t _offset;
    GGML_TYPE _ggml_type;

public:
    Tensor(const GGUFTensorInfo *info,
           const void *ggml_ptr,
           const GGUFKeyValue *shape_meta = nullptr,
           const GGUFKeyValue *strides_meta = nullptr,
           bool isOutput = false);
    Tensor(std::shared_ptr<Memory> memory, size_t offset,
           const std::vector<size_t> &shape,
           const std::vector<ptrdiff_t> &strides,
           GGML_TYPE dtype);
    ~Tensor();
    infiniopTensorDescriptor_t desc() const { return _desc; }
    std::vector<size_t> shape() const { return std::vector<size_t>(_shape); }
    std::vector<ptrdiff_t> strides() const { return std::vector<ptrdiff_t>(_strides); }
    GGML_TYPE ggml_type() const { return _ggml_type; }
    void *data() const;
    std::shared_ptr<Tensor> to(infiniDevice_t device, int device_id = 0) const;
    std::string info() const;
    void debug() const;
};
} // namespace infiniop_test

#endif
