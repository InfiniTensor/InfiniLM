#include "tensor.hpp"
#include "gguf.hpp"
#include "utils.hpp"
#include <cstring>
#include <infinirt.h>
#include <sstream>

template <typename T>
void printData(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            std::cout << *(data + i * strides[dim]) << " ";
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            printData(data + i * strides[dim], shape, strides, dim + 1);
            std::cout << std::endl;
        }
    }
}

// The type int8_t is represented by signed char, with a range of –128 to 127.
// It may contain non-printable characters and thus cannot be printed directly.
template <>
void printData(const int8_t *data, const std::vector<size_t> &shape,
               const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            std::cout << static_cast<int>(*(data + i * strides[dim])) << " ";
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            printData(data + i * strides[dim], shape, strides, dim + 1);
            std::cout << std::endl;
        }
    }
}

template <>
void printData(const bf16_t *data, const std::vector<size_t> &shape,
               const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            std::cout << utils::cast<float>(*(data + i * strides[dim])) << " ";
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            printData(data + i * strides[dim], shape, strides, dim + 1);
            std::cout << std::endl;
        }
    }
}

template <>
void printData(const fp16_t *data, const std::vector<size_t> &shape,
               const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            std::cout << utils::cast<float>(*(data + i * strides[dim])) << " ";
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            printData(data + i * strides[dim], shape, strides, dim + 1);
            std::cout << std::endl;
        }
    }
}

// Calculate memory size & offset given shape & strides
inline void calculateTensorMemory(size_t &size, size_t &offset,
                                  std::vector<size_t> shape,
                                  std::vector<ptrdiff_t> strides,
                                  size_t data_size) {
    size_t ndim = shape.size();
    offset = 0;
    size = 0;
    for (size_t i = 0; i < ndim; i++) {
        if (shape[i] == 0) {
            offset = 0;
            size = 0;
            return;
        }
        if (strides[i] > 0) {
            size += (shape[i] - 1) * strides[i] * data_size;
        } else if (strides[i] < 0) {
            offset += (shape[i] - 1) * (size_t)(-strides[i]) * data_size;
        }
    }
    size = offset + size + data_size;
}

namespace infiniop_test {

Memory::Memory(size_t size, infiniDevice_t device, int device_id) {
    _file_mapping = nullptr;
    _device = device;
    _device_id = device_id;
    _size = size;
    if (device == INFINI_DEVICE_CPU) {
        _ptr = std::malloc(size);
    } else {
        CHECK_OR(infinirtSetDevice(_device, _device_id), throw std::runtime_error("Error Creating Memory: set device"));
        CHECK_OR(infinirtMalloc(&_ptr, _size), throw std::runtime_error("Error Creating Memory: malloc"));
    }
}

Memory::Memory(const std::shared_ptr<FileMapping> &file_mapping, void *ptr, size_t size) {
    _device = INFINI_DEVICE_CPU;
    _device_id = 0;
    _size = size;
    _ptr = ptr;
    _file_mapping = file_mapping;
}

Memory::~Memory() {
    // if memory does not map to a file, free it manually
    if (_file_mapping == nullptr) {
        if (_device == INFINI_DEVICE_CPU) {
            std::free(_ptr);
        } else {
            infinirtSetDevice(_device, _device_id);
            infinirtFree(_ptr);
        }
    }
}

void *Tensor::data() const {
    return (char *)(_memory->ptr()) + _offset;
}

Tensor::Tensor(const GGUFTensorInfo *info,
               const void *ggml_ptr,
               const GGUFKeyValue *shape_meta,
               const GGUFKeyValue *strides_meta,
               bool isOutput) {

    _ggml_type = info->ggml_type;
    _offset = 0;
    size_t ndim = static_cast<size_t>(info->ndim);
    // `_shape`存储真实的tensor形状（来自shape_meta），`temp_shape`存储用于rearrange和计算内存的tensor形状
    _shape = std::vector<size_t>(ndim);
    std::vector<size_t> temp_shape(ndim);
    _strides = std::vector<ptrdiff_t>(ndim);
    std::vector<ptrdiff_t> contiguous_strides(ndim);
    for (size_t i = 0; i < ndim; i++) {
        temp_shape[i] = static_cast<size_t>(info->shape[ndim - 1 - i]);
        if (i == 0) {
            contiguous_strides[ndim - 1] = (ptrdiff_t)1;
        } else {
            contiguous_strides[ndim - 1 - i] = (ptrdiff_t)info->shape[i - 1] * contiguous_strides[ndim - i];
        }
        if (isOutput) {
            contiguous_strides[i] = (ptrdiff_t)0;
        }
    }

    if (strides_meta == nullptr) {
        for (size_t i = 0; i < ndim; i++) {
            _strides[i] = contiguous_strides[i];
        }
    } else {
        for (size_t i = 0; i < ndim; i++) {
            if (strides_meta->gguf_type == GGUF_TYPE_INT64) {
                _strides[i] = (ptrdiff_t)(reinterpret_cast<const int64_t *>(
                    strides_meta->value.data())[ndim - 1 - i]);
            } else if (strides_meta->gguf_type == GGUF_TYPE_INT32) {
                _strides[i] = (ptrdiff_t)(reinterpret_cast<const int32_t *>(
                    strides_meta->value.data())[ndim - 1 - i]);
            } else {
                throw std::runtime_error("Error Creating Tensor: Unsupported strides type");
            }
        }
    }

    if (isOutput) {
        if (shape_meta == nullptr) {
            throw std::runtime_error("Error Creating Tensor: shape_meta cannot be null for output tensor");
        }
        for (size_t i = 0; i < ndim; i++) {
            if (shape_meta->gguf_type == GGUF_TYPE_INT64) {
                int64_t val = reinterpret_cast<const int64_t *>(shape_meta->value.data())[i];
                if (val < 0) {
                    throw std::runtime_error("Shape must be non-negative");
                }
                temp_shape[i] = static_cast<size_t>(val);
            } else if (shape_meta->gguf_type == GGUF_TYPE_INT32) {
                int32_t val = reinterpret_cast<const int32_t *>(shape_meta->value.data())[i];
                if (val < 0) {
                    throw std::runtime_error("Shape must be non-negative");
                }
                temp_shape[i] = static_cast<size_t>(val);
            } else {
                throw std::runtime_error("Error Creating Tensor: Unsupported shape type");
            }
        }
    }
    infiniopCreateTensorDescriptor(&_desc, ndim, temp_shape.data(), _strides.data(), ggmlTypeToInfiniType(_ggml_type));
    size_t size;
    calculateTensorMemory(size, _offset, temp_shape, _strides, ggmlTypeSize(_ggml_type));
    _memory = std::make_shared<Memory>(size, INFINI_DEVICE_CPU, 0);
    utils::rearrange(
        (char *)_memory->ptr() + _offset,
        (char *)ggml_ptr + info->data_offset,
        temp_shape.data(),
        _strides.data(),
        contiguous_strides.data(),
        ndim,
        ggmlTypeSize(_ggml_type));

    if (shape_meta == nullptr) {
        _shape = temp_shape;
    } else {
        for (size_t i = 0; i < ndim; i++) {
            if (shape_meta->gguf_type == GGUF_TYPE_INT64) {
                int64_t val = reinterpret_cast<const int64_t *>(shape_meta->value.data())[i];
                if (val < 0) {
                    throw std::runtime_error("Shape must be non-negative");
                }
                _shape[i] = static_cast<size_t>(val);
            } else if (shape_meta->gguf_type == GGUF_TYPE_INT32) {
                int32_t val = reinterpret_cast<const int32_t *>(shape_meta->value.data())[i];
                if (val < 0) {
                    throw std::runtime_error("Shape must be non-negative");
                }
                _shape[i] = static_cast<size_t>(val);
            } else {
                throw std::runtime_error("Error Creating Tensor: Unsupported shape type");
            }
        }
    }
}

Tensor::Tensor(std::shared_ptr<Memory> memory, size_t offset,
               const std::vector<size_t> &shape,
               const std::vector<ptrdiff_t> &strides,
               GGML_TYPE dtype) : _memory(memory), _shape(shape), _strides(strides), _offset(offset), _ggml_type(dtype) {
    infiniopCreateTensorDescriptor(&_desc, shape.size(), shape.data(), strides.data(), ggmlTypeToInfiniType(dtype));
}

std::shared_ptr<Tensor> Tensor::to(infiniDevice_t device, int device_id) const {
    if (device == _memory->device() && (device_id == _memory->device_id() || device == INFINI_DEVICE_CPU)) {
        return std::make_shared<Tensor>(_memory, _offset, _shape, _strides, _ggml_type);
    }
    std::shared_ptr<Memory> memory;
    if (device == INFINI_DEVICE_CPU) {
        memory = std::make_shared<Memory>(_memory->size(), INFINI_DEVICE_CPU, 0);
        CHECK_OR(infinirtSetDevice(_memory->device(), _memory->device_id()), throw std::runtime_error("Error Tensor::to: set device"));
        CHECK_OR(infinirtMemcpy(memory->ptr(), _memory->ptr(), _memory->size(), INFINIRT_MEMCPY_D2H), throw std::runtime_error("Error Tensor::to: cpy"));
    } else if (_memory->device() == INFINI_DEVICE_CPU) {
        memory = std::make_shared<Memory>(_memory->size(), device, device_id);
        CHECK_OR(infinirtMemcpy(memory->ptr(), _memory->ptr(), _memory->size(), INFINIRT_MEMCPY_H2D), throw std::runtime_error("Error Tensor::to: cpy"));
    } else {
        return to(INFINI_DEVICE_CPU, 0)->to(device, device_id);
    }
    return std::make_shared<Tensor>(memory, _offset, _shape, _strides, _ggml_type);
}

void Tensor::debug() const {
    auto tensor = to(INFINI_DEVICE_CPU, 0);
    std::cout << "Tensor: " << tensor->info() << std::endl;
    switch (_ggml_type) {
    case GGML_TYPE_BF16:
        printData((bf16_t *)(tensor->data()), _shape, _strides, 0);
    case GGML_TYPE_F16:
        printData((fp16_t *)(tensor->data()), _shape, _strides, 0);
        break;
    case GGML_TYPE_F32:
        printData((float *)(tensor->data()), _shape, _strides, 0);
        break;
    case GGML_TYPE_F64:
        printData((double *)(tensor->data()), _shape, _strides, 0);
        break;
    case GGML_TYPE_Q8_K:
        printData((bool *)(tensor->data()), _shape, _strides, 0);
        break;
    case GGML_TYPE_I8:
        printData((int8_t *)(tensor->data()), _shape, _strides, 0);
        break;
    case GGML_TYPE_I16:
        printData((int16_t *)(tensor->data()), _shape, _strides, 0);
        break;
    case GGML_TYPE_I32:
        printData((int32_t *)(tensor->data()), _shape, _strides, 0);
        break;
    case GGML_TYPE_I64:
        printData((int64_t *)(tensor->data()), _shape, _strides, 0);
        break;
    default:
        std::cout << "Unsupported GGML type" << std::endl;
        break;
    }
}

std::string Tensor::info() const {
    std::ostringstream oss;
    oss << "Shape: [";
    for (size_t i = 0; i < _shape.size(); ++i) {
        oss << _shape[i];
        if (i != _shape.size() - 1) {
            oss << ", ";
        }
    }
    oss << "]";
    oss << ", Strides: [";
    for (size_t i = 0; i < _strides.size(); ++i) {
        oss << _strides[i];
        if (i != _strides.size() - 1) {
            oss << ", ";
        }
    }
    oss << "]";
    oss << ", Type: " << GGML_TYPE_NAME[_ggml_type];

    return oss.str();
}

Tensor::~Tensor() {
    infiniopDestroyTensorDescriptor(_desc);
}
} // namespace infiniop_test
