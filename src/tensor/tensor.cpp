#include "../tensor.hpp"
#include "../utils.hpp"
#include <fstream>
#include <iostream>
#include <numeric>

std::shared_ptr<TensorDesc>
TensorDesc::create(infiniDtype_t dtype, const std::vector<size_t> &shape,
                   const std::vector<ptrdiff_t> &strides) {
    auto desc = std::make_shared<TensorDesc>();
    infiniopCreateTensorDescriptor(&desc->_desc, shape.size(), shape.data(),
                                   strides.data(), dtype);
    return desc;
}

TensorDesc::~TensorDesc() {
    infiniopDestroyTensorDescriptor(this->_desc);
}

const std::vector<size_t> &Tensor::shape() const { return this->_shape; }
const std::vector<ptrdiff_t> &Tensor::strides() const { return this->_strides; }
size_t Tensor::ndim() const { return this->_shape.size(); }
infiniDtype_t Tensor::dtype() const { return this->_dtype; }
size_t Tensor::byte_size() const { return this->_size; }
infiniDevice_t Tensor::device_type() const { return this->storage->device_type; }
int Tensor::device_id() const { return this->storage->device_id; }
Tensor::~Tensor() {}

ptrdiff_t Tensor::data_offset() const {
    return (char *)(this->_data) - (char *)(this->storage->memory);
}

std::shared_ptr<TensorDesc> Tensor::desc() const { return TensorDesc::create(this->_dtype, this->_shape, this->_strides); }

std::shared_ptr<Tensor> Tensor::buffer(infiniDtype_t dtype,
                                       const std::vector<size_t> &shape,
                                       infinirtStream_t stream) {
    std::shared_ptr<Tensor> tensor = std::make_shared<Tensor>();
    tensor->_dtype = dtype;
    auto ndim = shape.size();
    if (shape.empty()) {
        tensor->_shape = std::vector<size_t>{1};
        ndim = 1;
    } else {
        tensor->_shape = std::vector<size_t>(shape);
    }
    size_t size = std::accumulate(shape.begin(), shape.end(), dsize(dtype), std::multiplies<size_t>());
    auto strides = std::vector<ptrdiff_t>(ndim);
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    tensor->_strides = strides;
    tensor->storage = Storage::createAsync(size, stream);
    tensor->_size = size;
    tensor->_data = tensor->storage->memory;
    infiniopCreateTensorDescriptor(&tensor->_desc, ndim, tensor->_shape.data(),
                                   strides.data(), dtype);
    tensor->_offset = 0;
    return tensor;
}

std::shared_ptr<Tensor> Tensor::weight(void *data, infiniDtype_t dtype,
                                       const std::vector<size_t> &shape) {
    std::shared_ptr<Tensor> tensor = std::make_shared<Tensor>();
    ;
    tensor->_dtype = dtype;
    auto ndim = shape.size();
    if (shape.empty()) {
        tensor->_shape = std::vector<size_t>{1};
        ndim = 1;
    } else {
        tensor->_shape = std::vector<size_t>(shape);
    }
    size_t size = std::accumulate(shape.begin(), shape.end(), dsize(dtype), std::multiplies<size_t>());
    auto strides = std::vector<ptrdiff_t>(ndim);
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    tensor->_strides = strides;
    tensor->storage = Storage::create(size);
    RUN_INFINI(infinirtMemcpy(tensor->storage->memory,
                              data, size, INFINIRT_MEMCPY_H2D));
    tensor->_data = tensor->storage->memory;
    tensor->_size = size;
    infiniopCreateTensorDescriptor(&tensor->_desc, ndim, tensor->_shape.data(),
                                   strides.data(), dtype);
    tensor->_offset = 0;
    return tensor;
}

void *Tensor::data_impl(ptrdiff_t offset) const {
    ASSERT(offset * dsize(this->dtype()) < this->_size);

    return (char *)(this->_data) + offset * dsize(this->dtype());
}

void *Tensor::data(ptrdiff_t offset) {
    return this->data_impl(offset);
}

const void *Tensor::data(ptrdiff_t offset) const {
    return this->data_impl(offset);
}

void Tensor::copy_from(std::shared_ptr<Tensor const> src,
                       infiniopHandle_t handle, infinirtStream_t stream) {
    ASSERT_EQ(this->shape(), src->shape());
    ASSERT_EQ(this->dtype(), src->dtype());
    infiniopRearrangeDescriptor_t desc;
    RUN_INFINI(infiniopCreateRearrangeDescriptor(
        handle, &desc, this->desc()->get(), src->desc()->get()));
    RUN_INFINI(infiniopRearrange(desc, this->data(), src->data(),
                                 stream));
    RUN_INFINI(infiniopDestroyRearrangeDescriptor(desc));
}

bool Tensor::is_contigous() const {
    auto ndim = this->ndim();
    auto shape = this->shape();
    auto strides = std::vector<ptrdiff_t>(ndim);
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    ASSERT_EQ(strides.size(), this->_strides.size());
    return std::equal(strides.begin(), strides.end(), this->_strides.begin());
}

template <typename T>
void print_data(T *data, const std::vector<size_t> &shape,
                const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            std::cout << data[i] << " ";
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
            std::cout << std::endl;
        }
    }
}

template <>
void print_data(uint16_t const *data, const std::vector<size_t> &shape,
                const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            std::cout << f16_to_f32(data[i * strides[dim]]) << " ";
        }
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
            std::cout << std::endl;
        }
    }
}

void Tensor::debug(const std::string &filename) const {
    RUN_INFINI(
        infinirtDeviceSynchronize());
    std::cout << "Tensor: "
              << "shape[ ";
    for (auto s : this->shape()) {
        std::cout << s << " ";
    }
    std::cout << "] strides[ ";
    for (auto s : this->strides()) {
        std::cout << s << " ";
    }
    std::cout << "] dtype=" << this->dtype()
              << " device=" << this->device_type()
              << " device_id=" << this->device_id() << std::endl;
    auto dtype = this->dtype();
    void const *cpu_data;
    if (this->device_type() != INFINI_DEVICE_CPU) {
        void *cpu_memory = std::malloc(this->storage->size);
        RUN_INFINI(infinirtMemcpy(cpu_memory, this->storage->memory,
                                  this->storage->size, INFINIRT_MEMCPY_D2H));
        cpu_data = cpu_memory;
    } else {
        cpu_data = this->data();
    }

    if (!filename.empty()) {
        std::ofstream outFile(filename, std::ios::binary);
        if (!outFile) {
            std::cerr << "Error opening file for writing: " << filename << "\n";
            return;
        }
        outFile.write(reinterpret_cast<const char *>(cpu_data), this->storage->size);
        outFile.close();
        std::cout << "Data written to file: " << filename << "\n";
        return;
    }

    switch (dtype) {
    case INFINI_DTYPE_F16:
        print_data((uint16_t const *)((char const *)cpu_data + data_offset()),
                   this->shape(), this->strides(), 0);
        break;
    case INFINI_DTYPE_F32:
        print_data((float const *)((char const *)cpu_data + data_offset()),
                   this->shape(), this->strides(), 0);
        break;
    case INFINI_DTYPE_U64:
        print_data((uint64_t const *)((char const *)cpu_data + data_offset()),
                   this->shape(), this->strides(), 0);
        break;
    case INFINI_DTYPE_I64:
        print_data((int64_t const *)((char const *)cpu_data + data_offset()),
                   this->shape(), this->strides(), 0);
        break;
    case INFINI_DTYPE_U32:
        print_data((uint32_t const *)((char const *)cpu_data + data_offset()),
                   this->shape(), this->strides(), 0);
        break;
    case INFINI_DTYPE_I32:
        print_data((int32_t const *)((char const *)cpu_data + data_offset()),
                   this->shape(), this->strides(), 0);
        break;
    default:
        PANIC("Unsupported data type");
    }
}

void Tensor::debug() const { this->debug(""); }
